def test_fedlora_exported_like_fedavg():
    from fedlearn.server import FedLoRA, FedAvg  # noqa: F401
    from collections import OrderedDict
    import torch
    s = FedLoRA(initial_parameters=OrderedDict([
        ("m.lora_A.weight", torch.zeros(1, 2)),
        ("m.lora_B.weight", torch.zeros(2, 1)),
    ]),
                aggregation="FFA_LORA")
    assert s.initialize_parameters() is not None
