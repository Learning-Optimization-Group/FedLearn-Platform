import grpc
from collections import OrderedDict
import torch
from typing import Tuple, Dict, Optional
import threading
import time

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.generated import fedlearn_pb2_grpc
from fedlearn.communication.serializer import proto_to_parameters, parameters_to_proto, parameters_to_chunks,USE_COMPRESSION

STREAMING_THRESHOLD_MB = 100
ALWAYS_STREAM_TRANSFORMERS = True
class GrpcClient:
    """A wrapper for the client-side gRPC functionality."""

    def __init__(self, client_id: str, server_address: str):
        self.client_id = client_id

        grpc_options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),

            # Keep connection alive aggressively
            ('grpc.keepalive_time_ms', 120000),  # Ping every 60 seconds
            ('grpc.keepalive_timeout_ms', 60000),  # Wait 5 seconds for pong
            ('grpc.keepalive_permit_without_calls', True),  # Ping even when idle
            ('grpc.http2.max_pings_without_data', 0),  # Allow unlimited pings
            ('grpc.http2.min_time_between_pings_ms', 120000),
            ('grpc.http2.min_ping_interval_without_data_ms', 120000),
            ('grpc.http2.bdp_probe', False),
            ('grpc.http2.max_ping_strikes', 0),

            # Increase connection timeouts
            ('grpc.max_connection_idle_ms', 7200000),  # 2 hours
            ('grpc.max_connection_age_ms', 14400000),  # 4 hours
            ('grpc.max_connection_age_grace_ms', 600000),
        ]

        # Channel for transferring parameters
        self.channel = grpc.insecure_channel(
            server_address,
            options=grpc_options
        )
        self.stub = fedlearn_pb2_grpc.FederatedLearningServiceStub(self.channel)

        # Parallel channel for keeping the Server-Client communication alive
        self.heartbeat_channel = grpc.insecure_channel(
            server_address,
            options=grpc_options
        )
        self.heartbeat_stub = fedlearn_pb2_grpc.FederatedLearningServiceStub(self.heartbeat_channel)

        self.heartbeat_active = False
        self.heartbeat_thread = None
        self.heartbeat_interval = 5  # seconds
        self.current_status = "idle"
        self.current_step = 0
        self.total_steps = 0
        self.current_round = 0

    def register(self) -> bool:
        """Registers the client with the server."""
        req = fedlearn_pb2.RegisterClientRequest(client_id=self.client_id)
        try:
            res = self.stub.RegisterClient(req)
            return res.status == fedlearn_pb2.RegisterClientResponse.Status.ACCEPTED
        except grpc.RpcError as e:
            print(f"ERROR: Could not register with server: {e.details()}")
            return False

    def get_global_model(self) -> Tuple[Optional[OrderedDict[str, torch.Tensor]], int, Dict]:
        """Fetches the latest global model using streaming to avoid memory issues."""
        req = fedlearn_pb2.GetGlobalModelRequest(client_id=self.client_id)

        try:
            self.update_status("downloading_model", 0, 0)

            # ALWAYS use streaming for large models
            print(f"[{self.client_id}] Downloading model using streaming...")

            chunks = []
            current_round = 0
            config = {}
            total_chunks = 0

            download_start = time.time()

            for chunk in self.stub.GetGlobalModelStream(req, timeout=3600):
                if chunk.chunk_index == 0:
                    current_round = chunk.current_round
                    config = dict(chunk.config)
                    total_chunks = chunk.total_chunks
                    print(f"[{self.client_id}] Receiving {total_chunks} chunk(s) for round {current_round}")

                chunks.append(chunk.chunk_data)
                progress = (chunk.chunk_index + 1) / chunk.total_chunks * 100

                if (chunk.chunk_index + 1) % 2 == 0 or chunk.is_final_chunk:  # Print every 2 chunks
                    print(
                        f"[{self.client_id}] Downloaded chunk {chunk.chunk_index + 1}/{chunk.total_chunks} ({progress:.1f}%)")

            download_time = time.time() - download_start
            print(f"[{self.client_id}] Download complete in {download_time:.1f}s")

            # Reconstruct model
            print(f"[{self.client_id}] Reconstructing model...")
            full_data = b''.join(chunks)

            import io
            buffer = io.BytesIO(full_data)
            model_data = torch.load(buffer, map_location='cpu', weights_only=False)
            buffer.close()

            params = model_data['parameters']

            self.current_round = current_round
            return params, current_round, config

        except grpc.RpcError as e:
            print(f"ERROR: Could not fetch global model: {e.details()}")
            raise e

    def _submit_update_unary(self, params: OrderedDict[str, torch.Tensor], num_examples: int,
                             round_number: int) -> bool:
        """Submit update using standard unary RPC (for small models)."""
        try:
            print(f"[{self.client_id}] Using standard unary upload...")

            params_proto = parameters_to_proto(params, num_examples)
            req = fedlearn_pb2.SubmitModelUpdateReque(
                client_id=self.client_id,
                parameters=params_proto,
                trained_on_round=round_number
            )

            res = self.stub.SubmitModelUpdate(req, timeout=300)
            return res.received

        except grpc.RpcError as e:
            print(f"ERROR: gRPC error during submit:")
            print(f"  Details: {e.details()}")
            print(f"  Status code: {e.code()}")
            print(f"  Debug error string: {e.debug_error_string()}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"ERROR: Exception during submit:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _submit_update_stream(self, params: OrderedDict[str, torch.Tensor], num_examples: int,
                              round_number: int) -> bool:
        """Submit update using client streaming RPC (for large models)."""
        try:
            print(f"[{self.client_id}] Using streaming upload...")

            upload_start = time.time()

            def chunk_generator():
                """Generate chunks for streaming."""
                try:
                    print(f"[{self.client_id}] Starting chunk generation...")
                    chunk_count = 0
                    for chunk_info in parameters_to_chunks(params, num_examples, compress=USE_COMPRESSION):
                        print(
                            f"[{self.client_id}] Creating chunk message {chunk_info['chunk_index'] + 1}/{chunk_info['total_chunks']}")

                        chunk_count += 1
                        chunk_start = time.time()
                        chunk_msg = fedlearn_pb2.ModelUpdateChunk(
                            client_id=self.client_id,
                            trained_on_round=round_number,
                            chunk_index=chunk_info['chunk_index'],
                            total_chunks=chunk_info['total_chunks'],
                            chunk_data=chunk_info['chunk_data'],
                            is_final_chunk=chunk_info['is_final_chunk'],
                            num_examples=chunk_info['num_examples']
                        )

                        # Progress update
                        progress = (chunk_info['chunk_index'] + 1) / chunk_info['total_chunks'] * 100
                        print(f"[{self.client_id}] Uploading chunk "
                              f"{chunk_info['chunk_index'] + 1}/{chunk_info['total_chunks']} ({progress:.1f}%)")

                        yield chunk_msg

                        chunk_time = time.time() - chunk_start
                        chunk_size_mb = len(chunk_info['chunk_data']) / (1024 ** 2)
                        speed_mbps = (chunk_size_mb * 8) / chunk_time if chunk_time > 0 else 0
                        print(f"[{self.client_id}] Chunk uploaded in {chunk_time:.1f}s ({speed_mbps:.2f} Mbps)")

                    print(f"[{self.client_id}] All chunks generated successfully")

                except Exception as e:
                    print(f"ERROR: Exception in chunk_generator: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            # Stream chunks to server
            print(f"[{self.client_id}] Starting gRPC stream...")
            response = self.stub.SubmitModelUpdateStream(chunk_generator(), timeout=3600)

            upload_time = time.time() - upload_start
            print(f"[{self.client_id}] Model update streamed successfully in {upload_time:.1f}s!")
            return response.received

        except grpc.RpcError as e:
            print(f"ERROR: Could not stream update: {e.details()}")
            print(f"ERROR: Status code: {e.code()}")
            print(f"ERROR: Debug error string: {e.debug_error_string()}")
            return False
        except Exception as e:
            print(f"ERROR: Exception during streaming: {e}")
            import traceback
            traceback.print_exc()
            return False

    def submit_update(self, params: OrderedDict[str, torch.Tensor], num_examples: int, round_number: int) -> bool:
        """Submit model update to server. Automatically chooses streaming for large models."""
        self.update_status("submitting_update", 0, 0)

        # Calculate model size
        total_params = sum(p.numel() for p in params.values())
        size_mb = (total_params * 4) / (1024 * 1024)  # float32
        size_gb = size_mb / 1024

        print(f"[{self.client_id}] Model: {size_gb:.2f} GB ({total_params:,} params, {size_mb:.2f} MB)")

        # Detect transformer models
        is_transformer = any(
            keyword in name.lower()
            for name in params.keys()
            for keyword in ['transformer', 'bert', 'gpt', 'opt', 'attention', 'encoder', 'decoder']
        )

        # Decision logic
        use_streaming = False
        reason = ""

        if is_transformer and ALWAYS_STREAM_TRANSFORMERS:
            use_streaming = True
            reason = "transformer model detected"
        elif size_mb > STREAMING_THRESHOLD_MB:
            use_streaming = True
            reason = f"size {size_mb:.2f} MB > threshold {STREAMING_THRESHOLD_MB} MB"

        if use_streaming:
            print(f"[{self.client_id}] Using STREAMING upload ({reason})")
            return self._submit_update_stream(params, num_examples, round_number)
        else:
            print(f"[{self.client_id}] Using STANDARD upload (size {size_mb:.2f} MB under threshold)")
            return self._submit_update_unary(params, num_examples, round_number)


    def send_heartbeat(self) -> bool:
        """Send a single heartbeat to the server."""
        req = fedlearn_pb2.HeartbeatRequest(
            client_id=self.client_id,
            status=self.current_status,
            current_step=self.current_step,
            total_steps=self.total_steps,
            current_round=self.current_round
        )

        try:
            res = self.heartbeat_stub.Heartbeat(req, timeout=30.0)

            if res.should_stop:
                print(f"[{self.client_id}] Server requested training stop")
                return False

            return res.acknowledged

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"[{self.client_id}] Heartbeat failed: Server unavailable")
                return False
            return False

    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeats."""
        import time
        while self.heartbeat_active:
            try:
                self.send_heartbeat()
            except Exception as e:
                # Heartbeat errors won't crash the thread
                pass
            time.sleep(self.heartbeat_interval)

    def start_heartbeat(self):
        """Start sending periodic heartbeats in background thread."""
        if not self.heartbeat_active:
            self.heartbeat_active = True
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True
            )
            self.heartbeat_thread.start()
            print(f"[{self.client_id}] Heartbeat started (every {self.heartbeat_interval}s)")

    def stop_heartbeat(self):
        """Stop sending heartbeats."""
        self.heartbeat_active = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2)
            print(f"[{self.client_id}] Heartbeat stopped")

    def update_status(self, status: str, current_step: int, total_steps: int):
        """Update current training status (will be sent in next heartbeat)."""
        self.current_status = status
        self.current_step = current_step
        self.total_steps = total_steps

    def close(self):
        """Closes the gRPC channel."""
        self.stop_heartbeat()
        self.channel.close()