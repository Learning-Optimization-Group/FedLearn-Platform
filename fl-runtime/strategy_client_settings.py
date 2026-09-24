"""Client-side training a first-order strategy's FL server asks for each round.

One source for both sides of the execution contract: fl_server.py hands these to FedOpt, which sends them to
every client each round, and execution_plan.py publishes them as the contract's local training. The rate a
contract states is therefore the rate the server sends. Torch-free so the resolver can import it cheaply.
"""
# FedOpt adapts on the server (FedAdam); its clients run plain SGD at this rate for this many epochs.
FEDOPT_CLIENT_LEARNING_RATE = 0.01
FEDOPT_CLIENT_LOCAL_EPOCHS = 1
