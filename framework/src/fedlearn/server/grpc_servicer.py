import hashlib
import logging
import time
from typing import List, Dict

import grpc
from concurrent import futures
import io
import torch
# Import the generated stubs
from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.generated import fedlearn_pb2_grpc

# The fedlearn.v2 protocol version this server speaks. Must equal the mobile client's kProtocolVersion
# (bridge/common/FedLearnCoreModule.h). RegisterClient rejects a mismatched client.
SERVER_PROTOCOL_VERSION = 2

# Import the business logic layer and helpers
from .coordinator import FLCoordinator, MalformedDeComFLSubmission
from fedlearn.communication.serializer import proto_to_parameters, parameters_to_proto, chunks_to_parameters
from fedlearn.server.decomfl_strategy import DeComFL


class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):
    """
    The gRPC servicer class. Acts as a dispatcher, forwarding calls to the FLCoordinator.
    """

    def __init__(self, coordinator: FLCoordinator):
        self.coordinator = coordinator

    def RegisterClient(self, request: fedlearn_pb2.RegisterClientRequest, context):
        client_id = request.client_id
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




            buffer = io.BytesIO()
            model_data = {'parameters': params, 'num_examples': 0}
            torch.save(model_data, buffer)
            data_to_send = buffer.getvalue()
            buffer.close()

            # FR-8 (download half): declare the sha256 of the FULL payload so receivers can
            # verify the reassembled blob. Set on EVERY chunk: the mobile C++ client reads it
            # from the first chunk (FedLearnClient.cpp), the Python client from the final one.
            # Purely additive — receivers that ignore the field behave exactly as before.
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
        client_id = "UNKNOWN"
        trained_on_round = -1

        try:
            client_id = request.client_id
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
        try:
            buffer = io.BytesIO()
            client_id = None
            round_num = None
            num_examples = 0
            total_chunks = 0
            chunks_received = 0

            logging.info(f"[Server] Receiving streamed model update...")

            # Stream chunks directly into buffer
            for chunk in request_iterator:
                if client_id is None:
                    client_id = chunk.client_id
                    round_num = chunk.trained_on_round
                    total_chunks = chunk.total_chunks
                    logging.info(f"[Server] Receiving {total_chunks} chunk(s) from {client_id} for round {round_num}")

                buffer.write(chunk.chunk_data)
                chunks_received += 1

                # Progress update
                progress = chunks_received / total_chunks * 100
                logging.info(f"[Server] Received chunk {chunks_received}/{total_chunks} ({progress:.1f}%)")

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
        if self.coordinator.stop_requested:
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
        try:
            client_id = request.client_id
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
                'num_perturbations': str(strategy.P)
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
        try:
            client_id = request.client_id
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
