from __future__ import annotations

﻿from typing import List, Dict

import grpc
from concurrent import futures
import io
import torch
# Import the generated stubs
from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.generated import fedlearn_pb2_grpc

# Import the business logic layer and helpers
from .coordinator import FLCoordinator
from fedlearn.communication.serializer import proto_to_parameters, parameters_to_proto, chunks_to_parameters, USE_COMPRESSION
from fedlearn.server.decomfl_strategy import DeComFL


class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):
    """
    The gRPC servicer class. Acts as a dispatcher, forwarding calls to the FLCoordinator.
    """

    def __init__(self, coordinator: FLCoordinator):
        self.coordinator = coordinator

    def RegisterClient(self, request: fedlearn_pb2.RegisterClientRequest, context):
        client_id = request.client_id
        success = self.coordinator.register_client(client_id)
        if success:
            return fedlearn_pb2.RegisterClientResponse(
                status=fedlearn_pb2.RegisterClientResponse.Status.ACCEPTED,
                message=f"Client '{client_id}' registered successfully."
            )
        else:  # In case registration logic becomes more complex
            return fedlearn_pb2.RegisterClientResponse(
                status=fedlearn_pb2.RegisterClientResponse.Status.REJECTED,
                message=f"Registration for '{client_id}' failed."
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

            print(f"[Server] Sending global model: {size_mb:.2f} MB")

            try:
                params_proto = parameters_to_proto(params, num_examples=0)
                return fedlearn_pb2.GetGlobalModelResponse(
                    parameters=params_proto,
                    current_round=current_round,
                    config=config
                )
            except MemoryError:
                print(f"[Server] MemoryError serializing {size_mb:.2f} MB model")
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                              f"Model too large ({size_mb:.2f} MB) for unary transfer. Client should use streaming.")


        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"An internal error occurred: {e}")
            import traceback
            traceback.print_exc()
            context.abort(grpc.StatusCode.INTERNAL, f"An internal error occurred: {str(e)}")

    def GetGlobalModelStream(self, request: fedlearn_pb2.GetGlobalModelRequest, context):
        """Stream global model to client for large models."""
        try:
            params, current_round, config = self.coordinator.get_global_model_for_client()

            if current_round == -1:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Training complete")

            if params is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Server not initialized")

            print(f"[Server] Streaming global model to {request.client_id} for round {current_round}")




            buffer = io.BytesIO()
            model_data = {'parameters': params, 'num_examples': 0}
            torch.save(model_data, buffer)
            data_to_send = buffer.getvalue()
            buffer.close()

            # Chunk the data
            chunk_size = 50 * 1024 * 1024  # 50 MB
            total_size = len(data_to_send)
            num_chunks = (total_size + chunk_size - 1) // chunk_size

            print(f"[Server] Sending {num_chunks} chunk(s) ({total_size / (1024 ** 2):.2f} MB)")

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
                    config=config if i == 0 else {}
                )

                if (i + 1) % 2 == 0 or (i == num_chunks - 1):
                    print(f"[Server] Sending chunk {i + 1}/{num_chunks}")

                yield chunk_msg

            print(f"[Server] Model stream complete")

        except Exception as e:
            print(f"[Server] Error streaming model: {e}")
            import traceback
            traceback.print_exc()
            context.abort(grpc.StatusCode.INTERNAL, f"Error: {str(e)}")

    def SubmitModelUpdate(self, request: fedlearn_pb2.SubmitModelUpdateReque, context):
        """Handle standard unary model update (for small models)."""
        client_id = "UNKNOWN"
        trained_on_round = -1

        try:
            client_id = request.client_id
            trained_on_round = request.trained_on_round

            print(f"=" * 60)
            print(f"[Server] SubmitModelUpdate START")
            print(f"[Server] Client: {client_id}")
            print(f"[Server] Round: {trained_on_round}")
            print(f"=" * 60)

            # Step 1: Deserialize parameters
            print(f"[Server] Step 1: Deserializing parameters...")
            params, num_examples = proto_to_parameters(request.parameters)
            print(f"[Server] Deserialized {len(params)} parameters")
            print(f"[Server] Num examples: {num_examples}")

            # Step 2: Submit to coordinator
            print(f"[Server] Step 2: Submitting to coordinator...")
            self.coordinator.submit_client_update(client_id, params, num_examples, trained_on_round)
            print(f"[Server] Coordinator accepted update")

            print(f"[Server] SubmitModelUpdate SUCCESS")
            print(f"=" * 60)
            return fedlearn_pb2.SubmitModelUpdateResponse(received=True)

        except Exception as e:
            # COMPREHENSIVE ERROR LOGGING
            print(f"!" * 60)
            print(f"[Server] CRITICAL ERROR in SubmitModelUpdate")
            print(f"[Server] Client: {client_id}")
            print(f"[Server] Round: {trained_on_round}")
            print(f"[Server] Error Type: {type(e).__name__}")
            print(f"[Server] Error Message: {str(e)}")
            print(f"!" * 60)

            # Full traceback
            import traceback
            traceback.print_exc()

            print(f"!" * 60)

            # Send detailed error to client
            error_msg = f"{type(e).__name__}: {str(e)}"
            context.abort(grpc.StatusCode.INTERNAL, error_msg)

    def SubmitModelUpdateStream(self, request_iterator, context):
        """
        Handle streamed model updates for large models.

        Args:
            request_iterator: Iterator of ModelUpdateChunk messages
            context: gRPC context

        Returns:
            SubmitModelUpdateResponse
        """
        try:
            chunks = []
            client_id = None
            round_num = None
            num_examples = 0
            total_chunks = 0

            print(f"[Server] Receiving streamed model update...")

            # Receive all chunks
            for chunk in request_iterator:
                if client_id is None:
                    client_id = chunk.client_id
                    round_num = chunk.trained_on_round
                    total_chunks = chunk.total_chunks
                    print(f"[Server] Receiving {total_chunks} chunk(s) from {client_id} for round {round_num}")

                chunks.append(chunk.chunk_data)

                # Progress update
                progress = len(chunks) / total_chunks * 100
                print(f"[Server] Received chunk {len(chunks)}/{total_chunks} ({progress:.1f}%)")

                if chunk.is_final_chunk:
                    num_examples = chunk.num_examples
                    break

            print(f"[Server] Received all {len(chunks)} chunk(s) from {client_id}")

            # Reconstruct parameters
            full_data = b''.join(chunks)
            print(f"[Server] Reconstructing model from {len(full_data) / (1024 ** 2):.2f} MB of data...")

            parameters, num_examples = chunks_to_parameters(full_data, compressed=USE_COMPRESSION)

            print(f"[Server] Model reconstructed successfully. Submitting to coordinator...")

            # Submit to coordinator
            self.coordinator.submit_client_update(client_id, parameters, num_examples, round_num)

            return fedlearn_pb2.SubmitModelUpdateResponse(received=True)

        except Exception as e:
            print(f"[Server] Error processing streamed update: {e}")
            import traceback
            traceback.print_exc()
            context.abort(grpc.StatusCode.INTERNAL, f"Error processing stream: {str(e)}")

    def GetServerStatus(self, request: fedlearn_pb2.GetServerStatusRequest, context):
        status = self.coordinator.get_server_status()
        return fedlearn_pb2.GetServerStatusResponse(
            server_state=fedlearn_pb2.GetServerStatusResponse.ServerState.WAITING_FOR_CLIENTS,  # Simplified for now
            current_round=status["current_round"],
            required_clients_for_round=status["required_clients_for_round"],
            received_updates_this_round=status["received_updates_this_round"]
        )

    def Heartbeat(self, request: fedlearn_pb2.HeartbeatRequest, context):
        """
        Handle heartbeat from client.
        This is a FAST call that doesn't block.
        """
        try:
            client_id = request.client_id
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
            print(f"ERROR: Heartbeat error for {request.client_id}: {e}")
            return fedlearn_pb2.HeartbeatResponse(
                acknowledged=False,
                should_stop=False,
                message=f"Error: {str(e)}"
            )

    def GetDeComFLConfig(self, request: fedlearn_pb2.GetDeComFLConfigRequest, context):
        """
        Handle DeComFL-specific configuration request.
        Returns seeds and rebuild history for the client.
        """
        try:
            client_id = request.client_id

            # Import here to avoid circular import
            from fedlearn.server.decomfl_strategy import DeComFL

            # Check if using DeComFL strategy
            if 'DeComFL' not in str(type(self.coordinator.strategy)):
                error_msg = "Server is not configured for DeComFL."
                print(f"[Server] ERROR: {error_msg}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(error_msg)
                return fedlearn_pb2.GetDeComFLConfigResponse()

            strategy = self.coordinator.strategy
            current_round = self.coordinator.current_round

            # Check if training is complete
            if self.coordinator.stop_requested:
                return fedlearn_pb2.GetDeComFLConfigResponse(current_round=-1)

            print(f"[Server] DeComFL config request from {client_id} for round {current_round}")

            # Generate seeds for current round
            seeds = strategy.generate_seeds(current_round)
            strategy.seed_history.append(seeds)

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

            print(f"[Server] Sending {len(seeds)} local steps, {len(rebuild_history)} missed rounds")

            return fedlearn_pb2.GetDeComFLConfigResponse(
                current_round=current_round,
                current_seeds=current_seeds_proto,
                rebuild_history=rebuild_history_proto,
                config=config
            )

        except Exception as e:
            print(f"[Server] Error in GetDeComFLConfig: {e}")
            print(f"[Server] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error: {str(e)}")
            return fedlearn_pb2.GetDeComFLConfigResponse()

    def SubmitGradientScalars(self, request: fedlearn_pb2.SubmitGradientScalarsRequest, context):
        """
        Handle submission of gradient scalars from DeComFL client.
        """
        try:
            client_id = request.client_id
            trained_on_round = request.trained_on_round
            num_examples = request.num_examples

            print(f"[Server] Receiving gradient scalars from {client_id} for round {trained_on_round}")

            # Import here to avoid circular import
            from fedlearn.server.decomfl_strategy import DeComFL

            # Check if using DeComFL strategy
            if 'DeComFL' not in str(type(self.coordinator.strategy)):
                error_msg = "Server is not configured for DeComFL."
                print(f"[Server] ERROR: {error_msg}")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(error_msg)
                return fedlearn_pb2.GetDeComFLConfigResponse()

            # Convert proto gradients to nested list format
            gradient_scalars = self._proto_to_gradients(request.gradients)

            print(f"[Server] Received {len(gradient_scalars)} local steps, "
                  f"{len(gradient_scalars[0]) if gradient_scalars else 0} perturbations per step")

            # Submit to coordinator (modified to handle DeComFL data)
            self.coordinator.submit_decomfl_update(
                client_id,
                gradient_scalars,
                num_examples,
                trained_on_round
            )

            print(f"[Server] Successfully received gradient scalars from {client_id}")

            return fedlearn_pb2.SubmitGradientScalarsResponse(received=True)

        except Exception as e:
            print(f"[Server] Error in SubmitGradientScalars: {e}")
            print(f"[Server] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error: {str(e)}")
            return fedlearn_pb2.SubmitGradientScalarsResponse(received=False)

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
