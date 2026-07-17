"""TE-15: smoke-verify the comparison harness end-to-end on the offline synthetic task."""
from benchmarks.algo_comparison import run_algorithm, decomfl_byte_projection, make_synthetic, rounds_to_target


def test_fedavg_run_improves_accuracy_and_accounts_bytes():
    out = run_algorithm("fedavg", make_synthetic(seed=0), num_clients=4, rounds=6, lr=0.1, seed=0)
    recs = out["records"]
    assert len(recs) == 6
    assert recs[-1]["accuracy"] >= recs[0]["accuracy"] - 1e-9  # non-degrading
    assert recs[-1]["accuracy"] > 0.6                          # the harness actually learns
    assert recs[0]["cum_bytes"] > 0                          # bytes measured
    assert recs[-1]["cum_bytes"] > recs[0]["cum_bytes"]      # monotone increasing


def test_fedprox_proximal_term_changes_the_trajectory():
    fa = run_algorithm("fedavg", make_synthetic(seed=1), num_clients=4, rounds=5, lr=0.1, seed=1)
    fp = run_algorithm("fedprox", make_synthetic(seed=1), num_clients=4, rounds=5, lr=0.1,
                       proximal_mu=1.0, seed=1)
    # A real proximal term makes FedProx a DIFFERENT trajectory than FedAvg (not a silent copy).
    fa_last, fp_last = fa["records"][-1], fp["records"][-1]
    assert (fa_last["accuracy"], fa_last["loss"]) != (fp_last["accuracy"], fp_last["loss"])


def test_fedopt_runs_and_produces_records():
    out = run_algorithm("fedopt", make_synthetic(seed=2), num_clients=4, rounds=4, lr=0.1, seed=2)
    assert len(out["records"]) == 4


def test_decomfl_byte_projection_is_orders_below_first_order():
    fa = run_algorithm("fedavg", make_synthetic(seed=0), num_clients=4, rounds=1, seed=0)
    dc = decomfl_byte_projection(fa["d"], rounds=1, num_clients=4)
    fo_one_round = fa["records"][0]["cum_bytes"]  # round-1 cum == one first-order round
    # >100x holds for realistic models (pinned in test_wire_bytes); the tiny synthetic model
    # still shows an order-of-magnitude win.
    assert dc["per_round_bytes"] < fo_one_round / 10
