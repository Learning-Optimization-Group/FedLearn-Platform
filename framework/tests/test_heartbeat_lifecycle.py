"""FR-10 — the heartbeat thread must stop promptly.

The loop used time.sleep(interval), so a stop request could not be observed until the sleep elapsed
(up to `heartbeat_interval` seconds), racing stop_heartbeat's join timeout and risking a leaked
thread. An Event-based wait lets stop() interrupt the wait immediately.
"""
from fedlearn.client.grpc_client import GrpcClient


def test_stop_heartbeat_exits_promptly_despite_a_long_interval():
    # 127.0.0.1:1 is unreachable, so each heartbeat RPC fails fast and the loop spends its time in the
    # inter-beat wait — exactly where a plain sleep would pin the thread until the interval elapsed.
    client = GrpcClient(client_id="hb", server_address="127.0.0.1:1")
    client.heartbeat_interval = 60
    try:
        client.start_heartbeat()
        assert client.heartbeat_thread.is_alive()

        client.stop_heartbeat()   # sets the stop event + joins with a short timeout internally

        assert not client.heartbeat_thread.is_alive(), \
            "heartbeat thread must exit on stop, not wait out the 60s interval"
    finally:
        client.close()
