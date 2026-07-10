import hashlib
import io
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import grpc
import torch

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.generated import fedlearn_pb2_grpc
from fedlearn.communication.serializer import (
    parameters_to_proto, parameters_to_chunks, chunks_to_parameters,
)
from fedlearn.security.client_interceptor import maybe_wrap_channel

log = logging.getLogger(__name__)

STREAMING_THRESHOLD_MB = 100
ALWAYS_STREAM_TRANSFORMERS = True

# Retryable gRPC status codes per https://grpc.io/docs/guides/status-codes/
_RETRYABLE_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}

T = TypeVar("T")


def _retry_unary(fn: Callable[[], T], *, op_name: str, max_attempts: int = 4,
                 base_delay: float = 0.5, max_delay: float = 8.0) -> T:
    """Retry a unary gRPC call on transient failures with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            attempt += 1
            if code not in _RETRYABLE_CODES or attempt >= max_attempts:
                log.error("%s failed (code=%s, attempt=%d): %s", op_name, code, attempt, e.details())
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            log.warning("%s transient failure (code=%s, attempt=%d); retrying in %.1fs",
                        op_name, code, attempt, delay)
            time.sleep(delay)


def _build_channel(server_address: str, grpc_options: list) -> grpc.Channel:
    """Builds a gRPC channel. Uses TLS when FEDLEARN_GRPC_USE_TLS=1."""
    use_tls = os.environ.get("FEDLEARN_GRPC_USE_TLS", "0") == "1"
    if not use_tls:
        return grpc.insecure_channel(server_address, options=grpc_options)

    root_cert_path = os.environ.get("FEDLEARN_GRPC_ROOT_CERT")
    client_cert_path = os.environ.get("FEDLEARN_GRPC_CLIENT_CERT")
    client_key_path = os.environ.get("FEDLEARN_GRPC_CLIENT_KEY")

    def _read(path: Optional[str]) -> Optional[bytes]:
        if not path:
            return None
        with open(path, "rb") as f:
            return f.read()

    credentials = grpc.ssl_channel_credentials(
        root_certificates=_read(root_cert_path),
        private_key=_read(client_key_path),
        certificate_chain=_read(client_cert_path),
    )
    return grpc.secure_channel(server_address, credentials, options=grpc_options)


class GrpcClient:
    """Client-side wrapper around the FederatedLearningService stub."""

    def __init__(self, client_id: str, server_address: str):
        self.client_id = client_id

        grpc_options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),

            # Keepalive tuned to survive AWS NLB / ALB idle-connection culling.
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 20000),
            ('grpc.keepalive_permit_without_calls', 1),
            ('grpc.http2.max_pings_without_data', 0),

            ('grpc.max_connection_idle_ms', 7200000),
            ('grpc.max_connection_age_ms', 14400000),
            ('grpc.max_connection_age_grace_ms', 600000),
        ]

        # SE-1: attach the connection token (from FEDLEARN_CONNECTION_TOKEN) to every call if present;
        # a no-op when unset, so dev / unauthenticated servers are unaffected.
        self.channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.stub = fedlearn_pb2_grpc.FederatedLearningServiceStub(self.channel)

        # Parallel channel for heartbeats so they don't contend with long transfers.
        self.heartbeat_channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.heartbeat_stub = fedlearn_pb2_grpc.FederatedLearningServiceStub(self.heartbeat_channel)

        self.heartbeat_active = False
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_interval = 5
        # FR-10: an interruptible wait so stop_heartbeat() can end the inter-beat delay immediately
        # instead of the thread having to sleep out the full interval before noticing it should stop.
        self._heartbeat_stop = threading.Event()
        # FR-10: server-driven stop. Latched by the heartbeat thread when a HeartbeatResponse
        # carries should_stop=True; the training thread's fit loop polls it between local steps
        # (should_stop_training()) and aborts the round. This is the cross-stub signal that lets
        # the parallel heartbeat stub halt a fit() blocking the training stub.
        self._stop_training = threading.Event()
        # Guards the status triple shared between the training thread (update_status writer)
        # and the heartbeat thread (snapshot reader) so a heartbeat never sees a torn write —
        # e.g. the status of one phase with the step count of another.
        self._status_lock = threading.Lock()
        self.current_status = "idle"
        self.current_step = 0
        self.total_steps = 0
        self.current_round = 0

    def register(self) -> bool:
        req = fedlearn_pb2.RegisterClientRequest(client_id=self.client_id)
        try:
            res = _retry_unary(lambda: self.stub.RegisterClient(req), op_name="RegisterClient")
            return res.status == fedlearn_pb2.RegisterClientResponse.Status.ACCEPTED
        except grpc.RpcError as e:
            log.error("register failed: %s", e.details())
            return False

    def get_global_model(self) -> Tuple[Optional[OrderedDict[str, torch.Tensor]], int, Dict]:
        """Fetches the latest global model via streaming."""
        req = fedlearn_pb2.GetGlobalModelRequest(client_id=self.client_id)

        try:
            self.update_status("downloading_model", 0, 0)
            log.info("[%s] Downloading model via streaming", self.client_id)

            # Stream chunks directly into a buffer to avoid 3x memory duplication.
            # Previously: chunks.append() + b''.join(chunks) + BytesIO(full_data)
            # allocated the payload three times. For large models (e.g. 14GB LLaMA-7B)
            # this caused OOM on edge devices like Jetson Orin.
            buffer = io.BytesIO()
            current_round = 0
            config: Dict[str, str] = {}
            total_chunks = 0
            codec = ""
            download_start = time.time()

            # FR-8 (download half): verify the server-declared sha256 of the full payload
            # before deserializing. Hash incrementally as chunks arrive so no second copy
            # of the payload is ever materialized (same OOM rationale as the BytesIO note
            # above). The server sets the hash on every chunk; accept it from whichever
            # chunk carries it (empty = pre-integrity server, verification skipped).
            hasher = hashlib.sha256()
            declared_sha256 = ""

            for chunk in self.stub.GetGlobalModelStream(req, timeout=3600):
                if chunk.chunk_index == 0:
                    current_round = chunk.current_round
                    config = dict(chunk.config)
                    total_chunks = chunk.total_chunks
                    codec = chunk.codec
                    log.info("[%s] Receiving %d chunk(s) for round %d",
                             self.client_id, total_chunks, current_round)

                if chunk.sha256:
                    declared_sha256 = chunk.sha256
                buffer.write(chunk.chunk_data)
                hasher.update(chunk.chunk_data)
                if (chunk.chunk_index + 1) % 2 == 0 or chunk.is_final_chunk:
                    progress = (chunk.chunk_index + 1) / chunk.total_chunks * 100
                    log.debug("[%s] Chunk %d/%d (%.1f%%)",
                              self.client_id, chunk.chunk_index + 1, chunk.total_chunks, progress)

            log.info("[%s] Download complete in %.1fs", self.client_id, time.time() - download_start)

            if declared_sha256:
                actual_sha256 = hasher.hexdigest()
                if actual_sha256 != declared_sha256:
                    buffer.close()
                    raise ValueError(
                        f"[{self.client_id}] Global model download failed sha256 integrity check: "
                        f"server declared {declared_sha256}, received payload hashes to "
                        f"{actual_sha256}. Refusing to deserialize."
                    )
                log.debug("[%s] Payload sha256 verified (%s...)", self.client_id, declared_sha256[:12])
            else:
                log.debug("[%s] Server declared no payload sha256 (pre-integrity server); "
                          "skipping verification", self.client_id)

            blob = buffer.getvalue()
            buffer.close()

            # FR-8 (download half) version gate: decode the SAFETENSORS wire (the current
            # format, symmetric with upload and the mobile C++ core) but transparently fall
            # back to a legacy torch.save pickle blob so a new client still works against an
            # old server during a staged rollout. The codec field is the primary signal; a
            # magic-byte sniff is the backstop when an old server sets no codec. Integrity
            # (sha256) was already verified above, format-agnostically.
            is_pickle = len(blob) >= 2 and (blob[:2] == b"PK" or blob[0] == 0x80)
            if codec.endswith("safetensors") if codec else not is_pickle:
                params, _num_examples = chunks_to_parameters(
                    blob, compressed=codec.startswith("lz4"))
            else:
                model_data = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
                params = model_data['parameters']

            self.current_round = current_round
            return params, current_round, config

        except grpc.RpcError as e:
            log.error("[%s] GetGlobalModel failed: %s", self.client_id, e.details())
            raise

    def _submit_update_unary(self, params: OrderedDict[str, torch.Tensor], num_examples: int,
                             round_number: int) -> bool:
        """Submit update using standard unary RPC (for small models)."""
        try:
            log.info("[%s] Unary model upload", self.client_id)
            params_proto = parameters_to_proto(params, num_examples)
            req = fedlearn_pb2.SubmitModelUpdateRequest(
                client_id=self.client_id,
                parameters=params_proto,
                trained_on_round=round_number,
            )
            res = _retry_unary(
                lambda: self.stub.SubmitModelUpdate(req, timeout=300),
                op_name="SubmitModelUpdate",
            )
            return res.received

        except grpc.RpcError as e:
            log.error("[%s] SubmitModelUpdate failed (code=%s): %s",
                      self.client_id, e.code(), e.details())
            return False
        except Exception:
            log.exception("[%s] Unexpected error in SubmitModelUpdate", self.client_id)
            return False

    def _generate_model_chunks(self, params: OrderedDict[str, torch.Tensor], num_examples: int,
                               round_number: int, chunk_size: int = 50 * 1024 * 1024):
        # Delegate serialization to parameters_to_chunks which now uses the safetensors wire
        # format (no lz4; the gRPC streaming path is uncompressed by design — see serializer.py).
        for chunk_dict in parameters_to_chunks(params, num_examples, chunk_size=chunk_size, compress=False):
            yield fedlearn_pb2.ModelUpdateChunk(
                client_id=self.client_id,
                trained_on_round=round_number,
                chunk_index=chunk_dict["chunk_index"],
                total_chunks=chunk_dict["total_chunks"],
                chunk_data=chunk_dict["chunk_data"],
                is_final_chunk=chunk_dict["is_final_chunk"],
                num_examples=num_examples,
            )

    def _submit_update_stream(self, params: OrderedDict[str, torch.Tensor], num_examples: int,
                              round_number: int) -> bool:
        """Submit update using client streaming RPC (for large models)."""
        try:
            log.info("[%s] Zero-copy streaming upload", self.client_id)
            upload_start = time.time()
            response = self.stub.SubmitModelUpdateStream(
                self._generate_model_chunks(params, num_examples, round_number),
                timeout=3600,
            )
            log.info("[%s] Streaming upload complete in %.1fs", self.client_id, time.time() - upload_start)
            return response.received

        except grpc.RpcError as e:
            log.error("[%s] SubmitModelUpdateStream failed (code=%s): %s",
                      self.client_id, e.code(), e.details())
            return False
        except Exception:
            log.exception("[%s] Unexpected error in SubmitModelUpdateStream", self.client_id)
            return False

    def submit_update(self, params: OrderedDict[str, torch.Tensor], num_examples: int, round_number: int) -> bool:
        """Submit model update. Auto-selects streaming for large / transformer models."""
        self.update_status("submitting_update", 0, 0)

        total_params = sum(p.numel() for p in params.values())
        size_mb = (total_params * 4) / (1024 * 1024)
        log.info("[%s] Model: %.2f MB (%s params)", self.client_id, size_mb, f"{total_params:,}")

        is_transformer = any(
            keyword in name.lower()
            for name in params.keys()
            for keyword in ['transformer', 'bert', 'gpt', 'opt', 'attention', 'encoder', 'decoder']
        )

        if (is_transformer and ALWAYS_STREAM_TRANSFORMERS) or size_mb > STREAMING_THRESHOLD_MB:
            reason = "transformer" if is_transformer else f"size {size_mb:.2f}MB > {STREAMING_THRESHOLD_MB}MB"
            log.info("[%s] Streaming upload selected (%s)", self.client_id, reason)
            return self._submit_update_stream(params, num_examples, round_number)

        log.info("[%s] Unary upload selected", self.client_id)
        return self._submit_update_unary(params, num_examples, round_number)

    def send_heartbeat(self) -> bool:
        status, current_step, total_steps, current_round = self._status_snapshot()
        req = fedlearn_pb2.HeartbeatRequest(
            client_id=self.client_id,
            status=status,
            current_step=current_step,
            total_steps=total_steps,
            current_round=current_round,
        )
        try:
            res = self.heartbeat_stub.Heartbeat(req, timeout=30.0)
            if res.should_stop:
                log.info("[%s] Server requested training stop", self.client_id)
                # FR-10: latch the stop so the training thread's fit loop can see it and
                # abort between local steps. Previously this response was discarded by
                # _heartbeat_loop, making the server's stop request a no-op.
                self._stop_training.set()
                return False
            return res.acknowledged
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                log.warning("[%s] Heartbeat: server unavailable", self.client_id)
            else:
                log.debug("[%s] Heartbeat failed: %s", self.client_id, e.details())
            return False

    def _heartbeat_loop(self):
        while self.heartbeat_active:
            try:
                self.send_heartbeat()
            except Exception:
                log.debug("[%s] Heartbeat loop exception", self.client_id, exc_info=True)
            # Interruptible: returns immediately once stop_heartbeat() sets the event (FR-10).
            self._heartbeat_stop.wait(self.heartbeat_interval)

    def start_heartbeat(self):
        if not self.heartbeat_active:
            self._heartbeat_stop.clear()
            self.heartbeat_active = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            log.info("[%s] Heartbeat started (every %ds)", self.client_id, self.heartbeat_interval)

    def stop_heartbeat(self):
        self.heartbeat_active = False
        self._heartbeat_stop.set()   # wake the loop out of its inter-beat wait at once
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
            log.info("[%s] Heartbeat stopped", self.client_id)

    def should_stop_training(self) -> bool:
        """True once the server has asked this client to abort training (via heartbeat).

        FR-10: polled by the fit loop between local steps on the TRAINING thread; set by the
        heartbeat thread. Once set it stays set — a server-driven stop ends the run.
        """
        return self._stop_training.is_set()

    def update_status(self, status: str, current_step: int, total_steps: int):
        """Publish the training thread's status triple for the heartbeat thread to report.

        Written under _status_lock so the paired _status_snapshot() read on the heartbeat
        thread can never observe a torn triple (three bare attribute stores are not atomic
        as a unit).
        """
        with self._status_lock:
            self.current_status = status
            self.current_step = current_step
            self.total_steps = total_steps

    def _status_snapshot(self) -> Tuple[str, int, int, int]:
        """Consistent (status, current_step, total_steps, current_round) for a heartbeat."""
        with self._status_lock:
            return self.current_status, self.current_step, self.total_steps, self.current_round

    def close(self):
        self.stop_heartbeat()
        self.channel.close()
        if hasattr(self, 'heartbeat_channel') and self.heartbeat_channel:
            self.heartbeat_channel.close()

    def get_server_status(self) -> Optional[fedlearn_pb2.GetServerStatusResponse]:
        """Best-effort fetch of the server's run/round state.

        Uses the heartbeat channel (separate from the training stub, which may be
        in a bad state after a cancelled call). Returns None if the server is
        unreachable — the caller treats "can't confirm" as "not complete".
        """
        try:
            req = fedlearn_pb2.GetServerStatusRequest()
            return self.heartbeat_stub.GetServerStatus(req, timeout=10.0)
        except grpc.RpcError as e:
            log.debug("[%s] GetServerStatus probe failed: %s", self.client_id, e.details())
            return None

    def server_reports_complete(self) -> bool:
        """True iff the server reports the run finished (rounds exhausted).

        This is the durable end-of-run signal: a completed run tears the gRPC
        RPCs down, so the client sees CANCELLED/UNAVAILABLE — but the server (if
        it is draining) still reports TRAINING_COMPLETE here, letting the client
        exit cleanly instead of retry-looping. A genuinely crashed/unreachable
        server returns None from the probe, so this returns False and the caller
        keeps its disconnect handling.
        """
        resp = self.get_server_status()
        if resp is None:
            return False
        return (
            resp.server_state
            == fedlearn_pb2.GetServerStatusResponse.ServerState.TRAINING_COMPLETE
        )

    def get_decomfl_config(self) -> Tuple[int, List[List[int]], List[Dict], dict]:
        """Fetch DeComFL configuration including seeds and rebuild history."""
        try:
            request = fedlearn_pb2.GetDeComFLConfigRequest(client_id=self.client_id)
            response = _retry_unary(
                lambda: self.stub.GetDeComFLConfig(request),
                op_name="GetDeComFLConfig",
            )

            if response.current_round == -1:
                return -1, [], [], {}

            seeds: List[List[int]] = []
            for local_step in response.current_seeds.local_steps:
                seeds.append(list(local_step.seeds))

            rebuild_history: List[Dict] = []
            for round_hist in response.rebuild_history.rounds:
                round_seeds: List[List[int]] = []
                for local_step in round_hist.seeds.local_steps:
                    round_seeds.append(list(local_step.seeds))

                round_grads: List[List[float]] = []
                for local_step in round_hist.average_gradients.local_steps:
                    round_grads.append(list(local_step.scalars))

                rebuild_history.append({
                    'round_number': round_hist.round_number,
                    'seeds': round_seeds,
                    'gradients': round_grads,
                })

            config = dict(response.config)
            return response.current_round, seeds, rebuild_history, config

        except grpc.RpcError as e:
            log.error("[%s] GetDeComFLConfig failed: %s", self.client_id, e.details())
            raise

    def submit_gradient_scalars(
            self,
            gradient_scalars: List[List[float]],
            num_examples: int,
            round_num: int,
    ) -> bool:
        """Submit gradient scalars instead of full model (DeComFL)."""
        try:
            local_steps = [
                fedlearn_pb2.LocalStepGradients(scalars=k_grads)
                for k_grads in gradient_scalars
            ]
            gradients_proto = fedlearn_pb2.GradientScalars(local_steps=local_steps)

            request = fedlearn_pb2.SubmitGradientScalarsRequest(
                client_id=self.client_id,
                trained_on_round=round_num,
                gradients=gradients_proto,
                num_examples=num_examples,
            )

            response = _retry_unary(
                lambda: self.stub.SubmitGradientScalars(request),
                op_name="SubmitGradientScalars",
            )
            return response.received

        except grpc.RpcError as e:
            log.error("[%s] SubmitGradientScalars failed: %s", self.client_id, e.details())
            return False
