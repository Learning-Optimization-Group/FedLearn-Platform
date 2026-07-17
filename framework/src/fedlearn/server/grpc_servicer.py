import hashlib
import logging
import os
import time
from typing import List, Dict

import grpc
from concurrent import futures
import io
import itertools
import torch
# Import the generated stubs
from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.generated import fedlearn_pb2_grpc

# The fedlearn.v2 protocol version this server speaks. Must equal the mobile client's kProtocolVersion
# (bridge/common/FedLearnCoreModule.h). RegisterClient rejects a mismatched client.
SERVER_PROTOCOL_VERSION = 2

# Import the business logic layer and helpers
from .coordinator import FLCoordinator, MalformedDeComFLSubmission
from fedlearn.communication.serializer import (
    proto_to_parameters, parameters_to_proto, chunks_to_parameters, state_dict_to_safetensors,
)
from fedlearn.server.decomfl_strategy import DeComFL

# SE-18: default caps on a single streamed model upload. Generous enough for LLM-scale adapters, but
# bounded so one client cannot grow the reassembly buffer without limit (bytes/chunks) or hold a
# server thread indefinitely (seconds). Override via env; a non-positive seconds value disables the
# wall-clock cap.
_DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 ** 3   # 2 GiB
_DEFAULT_MAX_UPLOAD_CHUNKS = 100_000
_DEFAULT_MAX_UPLOAD_SECONDS = 600.0         # 10 min of active streaming


class _StreamLimitExceeded(Exception):
    """SE-18: a streamed upload exceeded a resource cap (bytes/chunks -> RESOURCE_EXHAUSTED, or the
    wall-clock deadline -> DEADLINE_EXCEEDED). Carries the gRPC status code to abort with.

    A dedicated type (not ValueError/Exception) so the servicer's broad handlers below do not remap
    the abort to INVALID_ARGUMENT/INTERNAL — it is caught by its own clause first.
    """

    def __init__(self, message, code=grpc.StatusCode.RESOURCE_EXHAUSTED):
        super().__init__(message)
        self.code = code


class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):
    """
    The gRPC servicer class. Acts as a dispatcher, forwarding calls to the FLCoordinator.
    """

    def __init__(self, coordinator: FLCoordinator, partition_extractor=None):
        self.coordinator = coordinator
        # SE-15: (context) -> Optional[int] returning the verified connection-token partition; None
        # disables identity binding (client-auth off / dev fail-open), preserving existing behavior.
        self._partition_extractor = partition_extractor
        # SE-18: bound the streamed-upload reassembly buffer (memory-exhaustion DoS defense) and the
        # wall-clock time a single upload may spend actively streaming (slow-drip DoS defense).
        self._max_upload_bytes = int(os.environ.get("FEDLEARN_MAX_UPLOAD_BYTES", _DEFAULT_MAX_UPLOAD_BYTES))
        self._max_upload_chunks = int(os.environ.get("FEDLEARN_MAX_UPLOAD_CHUNKS", _DEFAULT_MAX_UPLOAD_CHUNKS))
        self._max_upload_seconds = float(os.environ.get("FEDLEARN_MAX_UPLOAD_SECONDS", _DEFAULT_MAX_UPLOAD_SECONDS))

    def _enforce_client_identity(self, client_id, context):
        """SE-15: pin one connection-token partition to one wire client_id. No-op when identity
        binding is disabled (auth off) or the call carries no verifiable token; otherwise aborts
        PERMISSION_DENIED when this client_id doesn't match the identity already bound to the token —
        stopping one valid token from being replayed under many client_ids to Sybil the cohort.

        MUST be called BEFORE any broad ``try/except`` in the RPC: ``context.abort`` raises, and that
        must reach gRPC rather than be swallowed as an INVALID_ARGUMENT/INTERNAL response.
        """
        if self._partition_extractor is None:
            return
        partition = self._partition_extractor(context)
        if partition is None:
            return
        if not self.coordinator.bind_or_check_identity(partition, client_id):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "client_id does not match the identity bound to this connection token",
            )

    def RegisterClient(self, request: fedlearn_pb2.RegisterClientRequest, context):
        client_id = request.client_id
        self._enforce_client_identity(client_id, context)  # SE-15: bind partition <-> client_id
        run_id = request.run_id
        client_pv = request.protocol_version
        # enrollment_token: minted by the Spring backend at enroll (P2). MVP validates permissively
        # (log-only) — a hard anti-Sybil check lands with the backend token endpoint.
        _enrollment_token = request.enrollment_token

        # Protocol-version negotiation (v2). A client that sends 0 (unset) is treated permissively;
        # a set-but-mismatched version is rejected so the two sides never silently disagree on the wire.
        if client_pv and client_pv != SERVER_PROTOCOL_VERSION:
            return fedlearn_pb2.RegisterClientResponse(
                status=fedlearn_pb2.RegisterClientResponse.Status.REJECTED,
                message=f"Protocol version mismatch: client={client_pv}, server={SERVER_PROTOCOL_VERSION}.",
                protocol_version=SERVER_PROTOCOL_VERSION,
            )

        success = self.coordinator.register_client(client_id)
        if success:
            return fedlearn_pb2.RegisterClientResponse(
                status=fedlearn_pb2.RegisterClientResponse.Status.ACCEPTED,
                message=f"Client '{client_id}' registered for run '{run_id}'.",
                assigned_round=self.coordinator.current_round,  # late joiners start at the live round
                protocol_version=SERVER_PROTOCOL_VERSION,
            )
        else:  # In case registration logic becomes more complex
            return fedlearn_pb2.RegisterClientResponse(
                status=fedlearn_pb2.RegisterClientResponse.Status.REJECTED,
                message=f"Registration for '{client_id}' failed.",
                protocol_version=SERVER_PROTOCOL_VERSION,
            )

    def GetGlobalModel(self, request: fedlearn_pb2.GetGlobalModelRequest, context):
        try:
            params, current_round, config = self.coordinator.get_global_model_for_client()

            # If the server is stopping, current_round will be -1
            if current_round == -1:
                return fedlearn_pb2.GetGlobalModelResponse(current_round=-1)

            if params is None:
                # If the server has not been initialized with a model yet
                context.abort(grpc.StatusCode.UNAVAILABLE, "Server is not yet initialized with a model. Please wait.")

            total_params = sum(p.numel() for p in params.values())
            size_mb = (total_params * 4) / (1024 * 1024)

            logging.info(f"[Server] Sending global model: {size_mb:.2f} MB")

            try:
                params_proto = parameters_to_proto(params, num_examples=0)
                return fedlearn_pb2.GetGlobalModelResponse(
                    parameters=params_proto,
                    current_round=current_round,
                    config=config
                )
            except MemoryError:
                logging.info(f"[Server] MemoryError serializing {size_mb:.2f} MB model")
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                              f"Model too large ({size_mb:.2f} MB) for unary transfer. Client should use streaming.")


        except Exception as e:
            logging.error(f"RPC failed for client {request.client_id}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "An internal server error occurred.")

    def GetGlobalModelStream(self, request: fedlearn_pb2.GetGlobalModelRequest, context):
        """Stream global model to client for large models."""
        try:
            params, current_round, config = self.coordinator.get_global_model_for_client()

            if current_round == -1:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Training complete")

            if params is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Server not initialized")

            logging.info(f"[Server] Streaming global model to {request.client_id} for round {current_round}")




            # FR-8 (download half): serialize the global model as a deterministic SAFETENSORS
            # blob — the same libtorch-free wire the upload path and the mobile C++ core already
            # use — instead of a torch.save pickle blob. F32-only and fail-loud (a non-float
            # param raises here rather than shipping a silently-cast/corrupt model). The mobile
            # FedLearnClient rejects any first chunk whose codec is not 'safetensors'; setting it
            # (with total_bytes) is what makes the FedAvg download decode instead of throw.
            data_to_send = state_dict_to_safetensors(params, num_examples=0)
            download_codec = "safetensors"

            # Declare the sha256 of the FULL payload so receivers can verify the reassembled
            # blob. Set on EVERY chunk: the mobile C++ client reads it from the first chunk
            # (FedLearnClient.cpp), the Python client from the final one. Integrity is
            # verified format-agnostically, before any deserialization.
            payload_sha256 = hashlib.sha256(data_to_send).hexdigest()

            # Chunk the data
            chunk_size = 50 * 1024 * 1024  # 50 MB
            total_size = len(data_to_send)
            num_chunks = (total_size + chunk_size - 1) // chunk_size

            logging.info(f"[Server] Sending {num_chunks} chunk(s) ({total_size / (1024 ** 2):.2f} MB, "
                         f"sha256={payload_sha256[:12]}...)")

            # Stream chunks
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, total_size)

                chunk_msg = fedlearn_pb2.ModelChunk(
                    chunk_index=i,
                    total_chunks=num_chunks,
                    chunk_data=data_to_send[start:end],
                    is_final_chunk=(i == num_chunks - 1),
                    current_round=current_round,
                    config=config if i == 0 else {},
                    codec=download_codec,
                    total_bytes=total_size,
                    sha256=payload_sha256
                )

                if (i + 1) % 2 == 0 or (i == num_chunks - 1):
                    logging.info(f"[Server] Sending chunk {i + 1}/{num_chunks}")

                yield chunk_msg

            logging.info(f"[Server] Model stream complete")

        except Exception as e:
            logging.error(f"RPC failed for client {request.client_id}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "An internal server error occurred.")

    def SubmitModelUpdate(self, request: fedlearn_pb2.SubmitModelUpdateRequest, context):
        """Handle standard unary model update (for small models)."""
        client_id = request.client_id
        self._enforce_client_identity(client_id, context)  # SE-15 (before the try: abort must reach gRPC)
        trained_on_round = -1

        try:
            trained_on_round = request.trained_on_round

            logging.info(f"=" * 60)
            logging.info(f"[Server] SubmitModelUpdate START")
            logging.info(f"[Server] Client: {client_id}")
            logging.info(f"[Server] Round: {trained_on_round}")
            logging.info(f"=" * 60)

            # Step 1: Deserialize parameters
            logging.info(f"[Server] Step 1: Deserializing parameters...")
            params, num_examples = proto_to_parameters(request.parameters)
            logging.info(f"[Server] Deserialized {len(params)} parameters")
            logging.info(f"[Server] Num examples: {num_examples}")

            # Step 2: Submit to coordinator
            logging.info(f"[Server] Step 2: Submitting to coordinator...")
            self.coordinator.submit_client_update(client_id, params, num_examples, trained_on_round)
            logging.info(f"[Server] Coordinator accepted update")

            logging.info(f"[Server] SubmitModelUpdate SUCCESS")
            logging.info(f"=" * 60)
            return fedlearn_pb2.SubmitModelUpdateResponse(received=True)

        except ValueError as e:
            # A malformed / non-finite payload is rejected by the serializer — a client error, so it
            # surfaces as INVALID_ARGUMENT (not the generic INTERNAL that hides the client's fault).
            logging.info(f"[Server] Rejecting invalid model update from {client_id}: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid model update: {e}")
        except Exception as e:
            logging.error(f"RPC failed for client {client_id}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "An internal server error occurred.")

    def SubmitModelUpdateStream(self, request_iterator, context):
        """
        Handle streamed model updates for large models.

        Uses direct BytesIO streaming to avoid 3x memory duplication from
        chunks.append() + b''.join(). See grpc_client.py for the same fix.

        Args:
            request_iterator: Iterator of ModelUpdateChunk messages
            context: gRPC context

        Returns:
            SubmitModelUpdateResponse
        """
        # SE-15: the client_id lives in the chunk stream, so resolve + enforce the identity BEFORE the
        # broad try/except below (whose `except Exception` would otherwise swallow the identity abort
        # as INTERNAL). Pull the first chunk, bind partition<->client_id, then feed it back into the
        # loop unchanged via itertools.chain.
        stream = iter(request_iterator)
        try:
            first_chunk = next(stream)
        except StopIteration:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty model update stream")
        self._enforce_client_identity(first_chunk.client_id, context)
        try:
            buffer = io.BytesIO()
            client_id = None
            round_num = None
            num_examples = 0
            total_chunks = 0
            chunks_received = 0
            total_bytes = 0  # SE-18: cumulative payload size, bounds-checked against the cap below
            # SE-18: wall-clock deadline for the active streaming loop. Checked on each chunk arrival,
            # so it bounds a slow-drip upload; a client that connects then goes fully silent blocks on
            # the next read and is bounded instead by gRPC's max_connection_age_ms / keepalive (set in
            # server.py). A non-positive cap disables this guard.
            deadline = time.monotonic() + self._max_upload_seconds

            logging.info(f"[Server] Receiving streamed model update...")

            # Stream chunks directly into buffer (the already-pulled first chunk is chained back in so
            # the header extraction below is unchanged).
            for chunk in itertools.chain([first_chunk], stream):
                if client_id is None:
                    client_id = chunk.client_id
                    round_num = chunk.trained_on_round
                    total_chunks = chunk.total_chunks
                    logging.info(f"[Server] Receiving {total_chunks} chunk(s) from {client_id} for round {round_num}")
                    # SE-18: reject an honestly-declared oversize upload up front, before buffering.
                    if chunk.total_bytes > self._max_upload_bytes:
                        raise _StreamLimitExceeded(
                            f"declared upload size {chunk.total_bytes} bytes exceeds the "
                            f"{self._max_upload_bytes}-byte cap (FEDLEARN_MAX_UPLOAD_BYTES)")

                # SE-18: bound the wall-clock time spent streaming this upload.
                if self._max_upload_seconds > 0 and time.monotonic() > deadline:
                    raise _StreamLimitExceeded(
                        f"streamed upload exceeded the {self._max_upload_seconds:.0f}s deadline "
                        f"(FEDLEARN_MAX_UPLOAD_SECONDS)", code=grpc.StatusCode.DEADLINE_EXCEEDED)

                # SE-18: enforce the caps BEFORE writing, so the buffer never exceeds the limit even if
                # the client lies about (or omits) total_bytes / total_chunks or never sends is_final.
                chunk_len = len(chunk.chunk_data)
                if total_bytes + chunk_len > self._max_upload_bytes:
                    raise _StreamLimitExceeded(
                        f"streamed upload exceeded the {self._max_upload_bytes}-byte cap "
                        f"(FEDLEARN_MAX_UPLOAD_BYTES)")
                if chunks_received + 1 > self._max_upload_chunks:
                    raise _StreamLimitExceeded(
                        f"streamed upload exceeded the {self._max_upload_chunks}-chunk cap "
                        f"(FEDLEARN_MAX_UPLOAD_CHUNKS)")

                buffer.write(chunk.chunk_data)
                total_bytes += chunk_len
                chunks_received += 1

                # Progress update. total_chunks is UNTRUSTED (SE-18: correctness rides on
                # is_final_chunk + the caps, never on this client-declared count), so a malformed
                # total_chunks <= 0 must not ZeroDivisionError here and get remapped to INTERNAL — the
                # percentage is advisory logging only.
                if total_chunks > 0:
                    progress = chunks_received / total_chunks * 100
                    logging.info(f"[Server] Received chunk {chunks_received}/{total_chunks} ({progress:.1f}%)")
                else:
                    logging.info(f"[Server] Received chunk {chunks_received} (total_chunks unset)")

                if chunk.is_final_chunk:
                    num_examples = chunk.num_examples
                    break

            logging.info(f"[Server] Received all {chunks_received} chunk(s) from {client_id}")

            # Reconstruct parameters from the streamed buffer
            buffer.seek(0)
            full_data = buffer.read()
            buffer.close()
            logging.info(f"[Server] Reconstructing model from {len(full_data) / (1024 ** 2):.2f} MB of data...")

            # Streamed uploads are raw (uncompressed), matching the client producer
            # (_generate_model_chunks) and the download path (GetGlobalModelStream). Decompression
            # is NOT keyed off the per-process FEDLEARN_USE_COMPRESSION env var, which the client
            # and this server-spawning backend do not necessarily share.
            parameters, num_examples = chunks_to_parameters(full_data, compressed=False)

            logging.info(f"[Server] Model reconstructed successfully. Submitting to coordinator...")

            # Submit to coordinator
            self.coordinator.submit_client_update(client_id, parameters, num_examples, round_num)

            return fedlearn_pb2.SubmitModelUpdateResponse(received=True)

        except _StreamLimitExceeded as e:
            # SE-18: a size cap (RESOURCE_EXHAUSTED) or the wall-clock deadline (DEADLINE_EXCEEDED)
            # was hit. Caught BEFORE the broad handlers so the abort isn't remapped to
            # INVALID_ARGUMENT/INTERNAL; the exact code rides on the exception.
            logging.warning(f"[Server] Rejecting streamed upload from {client_id}: {e}")
            context.abort(e.code, str(e))
        except ValueError as e:
            # Malformed / non-finite streamed payload -> client error -> INVALID_ARGUMENT.
            logging.info(f"[Server] Rejecting invalid streamed model update from {client_id}: {e}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid model update: {e}")
        except Exception as e:
            logging.error(f"RPC failed for client {client_id}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, "An internal server error occurred.")

    def GetServerStatus(self, request: fedlearn_pb2.GetServerStatusRequest, context):
        status = self.coordinator.get_server_status()
        active = len(self.coordinator.get_active_clients())
        State = fedlearn_pb2.GetServerStatusResponse.ServerState
        # Report the terminal state so a client can distinguish a finished run (exit cleanly) from a
        # transient failure (keep retrying). training_complete is the precise "all rounds done"
        # signal; stop_requested also covers the abort path.
        if status.get("training_complete") or self.coordinator.stop_requested:
            server_state = State.TRAINING_COMPLETE
        elif active < status["required_clients_for_round"]:
            server_state = State.WAITING_FOR_CLIENTS
        else:
            server_state = State.TRAINING
        # A rolling deadline (now + per-round timeout) so a client's status poll never implies an
        # infinite wait (v2 §6.2). A precise per-round start-stamp can replace this post-MVP.
        round_deadline_unix_ms = int((time.time() + self.coordinator.round_timeout_s) * 1000)
        return fedlearn_pb2.GetServerStatusResponse(
            server_state=server_state,
            current_round=status["current_round"],
            required_clients_for_round=status["required_clients_for_round"],
            received_updates_this_round=status["received_updates_this_round"],
            active_clients=active,
            round_deadline_unix_ms=round_deadline_unix_ms,
        )

    def Heartbeat(self, request: fedlearn_pb2.HeartbeatRequest, context):
        """
        Handle heartbeat from client.
        This is a FAST call that doesn't block.
        """
        client_id = request.client_id
        self._enforce_client_identity(client_id, context)  # SE-15 (before the try: abort must reach gRPC)
        try:
            run_id = request.run_id  # v2 field 2 — the run this heartbeat belongs to
            status = request.status
            current_step = request.current_step
            total_steps = request.total_steps
            current_round = request.current_round

            acknowledged, should_stop, message = self.coordinator.update_client_heartbeat(
                client_id, status, current_step, total_steps, current_round
            )

            return fedlearn_pb2.HeartbeatResponse(
                acknowledged=acknowledged,
                should_stop=should_stop,
                message=message
            )

        except Exception as e:
            logging.error(f"RPC failed for client {request.client_id}", exc_info=True)
            return fedlearn_pb2.HeartbeatResponse(
                acknowledged=False,
                should_stop=False,
                message="An internal server error occurred."
            )

    def GetDeComFLConfig(self, request: fedlearn_pb2.GetDeComFLConfigRequest, context):
        """
        Handle DeComFL-specific configuration request.
        Returns seeds and rebuild history for the client.
        """
        try:
            client_id = request.client_id

            # FR-6: typed dispatch (DeComFL is imported at module level — no circular import).
            if not isinstance(self.coordinator.strategy, DeComFL):
                error_msg = "Server is not configured for DeComFL."
                logging.info(f"[Server] ERROR: {error_msg}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(error_msg)
                return fedlearn_pb2.GetDeComFLConfigResponse()

            strategy = self.coordinator.strategy
            current_round = self.coordinator.current_round

            # Check if training is complete
            if self.coordinator.stop_requested:
                return fedlearn_pb2.GetDeComFLConfigResponse(current_round=-1)

            logging.info(f"[Server] DeComFL config request from {client_id} for round {current_round}")

            # Get the seeds for this round — generated ONCE and shared by every client
            # (audit #28: previously regenerated + list-appended per client RPC, which gave each
            # client a different perturbation direction and corrupted seed_history indexing).
            seeds = strategy.get_or_create_seeds(current_round)

            # Get rebuild history for missed rounds
            rebuild_history = strategy.get_rebuild_history(client_id, current_round)

            # Convert to proto format
            current_seeds_proto = self._seeds_to_proto(seeds)
            rebuild_history_proto = self._rebuild_history_to_proto(rebuild_history)

            # Configuration
            config = {
                'learning_rate': str(strategy.eta),
                'smoothing_param': str(strategy.mu),
                'num_local_steps': str(strategy.K),
                'num_perturbations': str(strategy.P),
                # MO-19/FR-14: advertise the server's trainable flat dimension so every client (python
                # or mobile) can fail loud at the handshake if its own trainable dim differs — instead
                # of training on a misaligned shared-seed perturbation and diverging silently.
                'model_dim': str(strategy.model_dim),
            }

            logging.info(f"[Server] Sending {len(seeds)} local steps, {len(rebuild_history)} missed rounds")

            return fedlearn_pb2.GetDeComFLConfigResponse(
                current_round=current_round,
                current_seeds=current_seeds_proto,
                rebuild_history=rebuild_history_proto,
                config=config,
                # v2 determinism contract. The mobile core is RNG-version-independent (RandnEngine byte-
                # matches torch.randn regardless of torch build), so torch_version is advisory; the client
                # does not gate on it. grad_estimate_method mirrors the strategy (forward-difference).
                # golden_vector_sha256 is an optional RNG-parity fixture (empty ⇒ the client skips the check).
                torch_version=torch.__version__,
                grad_estimate_method=getattr(strategy, "grad_estimate_method", "forward"),
                golden_vector_sha256="",
            )

        except Exception as e:
            logging.error(f"RPC failed for client {request.client_id}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An internal server error occurred.")
            return fedlearn_pb2.GetDeComFLConfigResponse()

    def SubmitGradientScalars(self, request: fedlearn_pb2.SubmitGradientScalarsRequest, context):
        """
        Handle submission of gradient scalars from DeComFL client.
        """
        client_id = request.client_id
        self._enforce_client_identity(client_id, context)  # SE-15 (before the try: abort must reach gRPC)
        try:
            trained_on_round = request.trained_on_round
            num_examples = request.num_examples

            logging.info(f"[Server] Receiving gradient scalars from {client_id} for round {trained_on_round}")

            # FR-6: typed dispatch (DeComFL is imported at module level — no circular import).
            if not isinstance(self.coordinator.strategy, DeComFL):
                error_msg = "Server is not configured for DeComFL."
                logging.info(f"[Server] ERROR: {error_msg}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(error_msg)
                # FR-6: return THIS RPC's response type, not GetDeComFLConfigResponse.
                return fedlearn_pb2.SubmitGradientScalarsResponse(received=False)

            # Convert proto gradients to nested list format
            gradient_scalars = self._proto_to_gradients(request.gradients)

            # v2: the DeComFL client echoes the server-issued seeds in perturbation_seeds. The server
            # reconstructs z from its own shared seed_history (get_or_create_seeds), so the echo is
            # advisory here (observability / a future integrity cross-check); it is NOT re-derived from.
            echoed_steps = len(request.perturbation_seeds.local_steps) if request.HasField("perturbation_seeds") else 0

            logging.info(f"[Server] Received {len(gradient_scalars)} local steps, "
                  f"{len(gradient_scalars[0]) if gradient_scalars else 0} perturbations per step "
                  f"(echoed seed steps: {echoed_steps})")

            # Submit to coordinator (modified to handle DeComFL data)
            self.coordinator.submit_decomfl_update(
                client_id,
                gradient_scalars,
                num_examples,
                trained_on_round
            )

            logging.info(f"[Server] Successfully received gradient scalars from {client_id}")

            return fedlearn_pb2.SubmitGradientScalarsResponse(received=True)

        except MalformedDeComFLSubmission as e:
            # FR-5: a wrong-shaped grid is the client's fault, not a server error — surface it as
            # INVALID_ARGUMENT so the client can correct the payload, rather than an opaque INTERNAL.
            logging.info(f"[Server] Rejecting malformed DeComFL submit from {request.client_id}: {e}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return fedlearn_pb2.SubmitGradientScalarsResponse(received=False)

        except Exception as e:
            logging.error(f"RPC failed for client {request.client_id}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An internal server error occurred.")
            return fedlearn_pb2.SubmitGradientScalarsResponse(received=False)

    def ReportClientMetrics(self, request: fedlearn_pb2.ReportClientMetricsRequest, context):
        """v2 telemetry (§6.4): accept a client's per-round loss/accuracy/compute and record it on the
        coordinator for the dashboard. Best-effort — a telemetry failure never fails the round."""
        try:
            self.coordinator.record_client_metrics({
                "client_id": request.client_id,
                "run_id": request.run_id,
                "round": request.round,
                "loss": request.loss,
                "accuracy": request.accuracy,
                "current_step": request.current_step,
                "total_steps": request.total_steps,
                "client_type": request.client_type,
                "compute_ms": request.compute_ms,
            })
            logging.info(
                f"[Server] Metrics from {request.client_id} round {request.round}: "
                f"loss={request.loss:.4f} acc={request.accuracy:.4f} ({request.compute_ms}ms)"
            )
            return fedlearn_pb2.ReportClientMetricsResponse(acknowledged=True)
        except Exception:
            logging.error(f"ReportClientMetrics failed for client {request.client_id}", exc_info=True)
            return fedlearn_pb2.ReportClientMetricsResponse(acknowledged=False)

    # Helper methods for proto conversion
    def _seeds_to_proto(self, seeds: List[List[int]]) -> fedlearn_pb2.PerturbationSeeds:
        """Convert nested list of seeds to proto format."""
        local_steps = []
        for k_seeds in seeds:
            local_step_seeds = fedlearn_pb2.LocalStepSeeds(seeds=k_seeds)
            local_steps.append(local_step_seeds)
        return fedlearn_pb2.PerturbationSeeds(local_steps=local_steps)

    def _rebuild_history_to_proto(self, history: List[Dict]) -> fedlearn_pb2.RebuildHistory:
        """Convert rebuild history to proto format."""
        rounds = []
        for round_data in history:
            seeds_proto = self._seeds_to_proto(round_data['seeds'])
            gradients_proto = self._gradients_to_proto(round_data['gradients'])

            round_history = fedlearn_pb2.RoundHistory(
                round_number=round_data['round_number'],
                seeds=seeds_proto,
                average_gradients=gradients_proto
            )
            rounds.append(round_history)

        return fedlearn_pb2.RebuildHistory(rounds=rounds)

    def _gradients_to_proto(self, gradients: List[List[float]]) -> fedlearn_pb2.GradientScalars:
        """Convert nested list of gradient scalars to proto format."""
        local_steps = []
        for k_grads in gradients:
            local_step_grads = fedlearn_pb2.LocalStepGradients(scalars=k_grads)
            local_steps.append(local_step_grads)
        return fedlearn_pb2.GradientScalars(local_steps=local_steps)

    def _proto_to_gradients(self, proto: fedlearn_pb2.GradientScalars) -> List[List[float]]:
        """Convert proto gradient scalars to nested list format."""
        gradients = []
        for local_step in proto.local_steps:
            gradients.append(list(local_step.scalars))
        return gradients
