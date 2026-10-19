from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from fedlearn.communication.generated import tree_pb2 as p, tree_pb2_grpc as rpc
from fedlearn.communication.generated import fedlearn_pb2 as neural, fedlearn_pb2_grpc as neural_rpc
from fedlearn.tree.coordinator import TreeCoordinator
from fedlearn.tree.transport import add_tree_services
from test_rounds import contract, request


def test_real_grpc_refuses_old_protocol_and_deduplicates_valid_tree_updates():
    value, initial = contract(); coordinator = TreeCoordinator(value,initial)
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    add_tree_services(server,coordinator,partition_extractor=lambda context:
                      int(dict(context.invocation_metadata()).get("test-partition","-1")))
    port=server.add_insecure_port("127.0.0.1:0"); server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            old=neural_rpc.FederatedLearningServiceStub(channel)
            response=old.RegisterClient(neural.RegisterClientRequest(client_id="old",run_id=value.run_id,protocol_version=2))
            assert response.status==neural.RegisterClientResponse.REJECTED
            tree=rpc.FederatedTreeServiceStub(channel)
            for client,partition in (("a",0),("b",1)):
                registration=p.TreeRegistration(client_id=client,run_id=value.run_id,
                    contract_id=coordinator.contract_id,protocol_version=3,partition_id=partition)
                assert tree.RegisterClient(registration,metadata=[("test-partition",str(partition))]).protocol_version==3
            with pytest.raises(grpc.RpcError):
                tree.GetTreeModel(p.TreeModelRequest(client_id="a",run_id=value.run_id,
                    contract_id=coordinator.contract_id),metadata=[("test-partition","1")])
            a=request(coordinator,"a")
            assert tree.SubmitTreeUpdate(a,metadata=[("test-partition","0")]).result==p.TreeContributionAck.ACCEPTED
            assert tree.SubmitTreeUpdate(a,metadata=[("test-partition","0")]).result==p.TreeContributionAck.ALREADY_ACCEPTED
    finally:
        server.stop(0).wait()
