"""Tree RPC dispatch with enrollment binding and explicit neural refusal."""
import grpc

from fedlearn.communication.generated import tree_pb2 as p, tree_pb2_grpc as rpc
from fedlearn.communication.generated import fedlearn_pb2 as neural, fedlearn_pb2_grpc as neural_rpc


class TreeService(rpc.FederatedTreeServiceServicer):
    def __init__(self, coordinator, partition_extractor=None):
        self.coordinator = coordinator
        self.partition_extractor = partition_extractor

    def _partition(self, request, context, registration=False):
        coordinator = self.coordinator
        if request.run_id != coordinator.contract.run_id or request.contract_id != coordinator.contract_id:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "tree run or contract mismatch")
        if self.partition_extractor:
            partition = self.partition_extractor(context)
            if partition is None or (registration and partition != request.partition_id):
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "missing or mismatched enrollment partition")
        else:
            # Explicitly unauthenticated development mode only.
            partition = request.partition_id if registration else coordinator.clients.get(request.client_id)
        if not registration:
            try:
                coordinator.check_identity(request.client_id, partition)
            except ValueError as exc:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        return partition

    def RegisterClient(self, request, context):
        if request.protocol_version != 3:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "tree clients require protocol 3")
        partition = self._partition(request, context, registration=True)
        try:
            self.coordinator.register(request.client_id, partition=partition)
        except ValueError as exc:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        except TimeoutError as exc:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))
        return p.TreeRegistrationAck(protocol_version=3)

    def GetTreeModel(self, request, context):
        with self.coordinator.condition:
            self._partition(request, context)
            try:
                return self.coordinator.snapshot()
            except TimeoutError as exc:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))

    def SubmitTreeUpdate(self, request, context):
        with self.coordinator.condition:
            self._partition(request, context)
            try:
                result = self.coordinator.submit(request)
            except ValueError as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except TimeoutError as exc:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))
            return p.TreeContributionAck(result=result)


class RefuseNeural(neural_rpc.FederatedLearningServiceServicer):
    def RegisterClient(self, request, context):
        return neural.RegisterClientResponse(status=neural.RegisterClientResponse.REJECTED,
            message="This run trains XGBoost trees; use a protocol-3 tree client", protocol_version=3)


def _refuse(request, context):
    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "This run supports tree RPCs only")


for _method in neural.DESCRIPTOR.services_by_name["FederatedLearningService"].methods:
    if _method.name != "RegisterClient":
        setattr(RefuseNeural, _method.name, _refuse)


def add_tree_services(server, coordinator, partition_extractor=None):
    rpc.add_FederatedTreeServiceServicer_to_server(TreeService(coordinator, partition_extractor), server)
    neural_rpc.add_FederatedLearningServiceServicer_to_server(RefuseNeural(), server)
