# Production central DP: which neighbour relation the ε covers

22 October 2026. This covers shortlist item 1 of the
[literature check](literature-review-robust-private-fl-2026-10-22.md). It is analysis and tests only;
no behaviour has changed. The tests are in `framework/tests/test_dp_adjacency.py`.

## What the code does

`dp_aggregate` serves both FedLoRA DP and `DPFedAvg`. It:
1. clips each client's delta jointly to L2 norm S;
2. takes the uniform mean over the n clients it received;
3. adds Gaussian noise with std z·S/n per coordinate.

The accountant (RDP, q = 1) treats z as noise std divided by L2 sensitivity. So the reported ε is
correct exactly when the released mean moves by at most S/n between neighbouring inputs.

On the live path, n is always the fixed, public cohort size:
- `require_live_dp_accounting` refuses a DP run unless `--clients-per-round` equals `--min-clients`.
- The coordinator aggregates only when exactly `clients_per_round` updates have arrived
  (`coordinator.py:543`).
- A timeout force-aggregates only if at least `min_clients` reported, and on a DP run that is all of them.

## How far the mean moves (pinned by the tests)

| Neighbour relation | Max shift of the mean | Effective multiplier |
| --- | --- | --- |
| Fixed cohort, one client's delta replaced by **zero** (add/remove with a fixed public denominator) | S/n | z — as accounted |
| Fixed cohort, one client's delta replaced by **any other** clipped delta (replace-one) | 2S/n | z/2 |
| Cohort size allowed to change, mean over the actual count (add/remove with a data-dependent denominator) | ≈ 2S/n | ≈ z/2 |

## Verdict

**This is not a bug in the narrow sense; it is an unstated, weaker guarantee.**

- The reported ε is valid for the first relation: an observer cannot tell whether a given client
  contributed its update or contributed nothing. This is the convention of standard DP-FedAvg
  (McMahan et al., ICLR'18) with a fixed denominator. The live path qualifies because n is always the
  fixed cohort size.
- It is not valid for replace-one ("cannot tell this client's data from any other client's"), which is
  the relation the capstone simulator uses. For the same noise, the replace-one ε is much larger:

| Claimed ε (RDP, add/remove) | z (T = 20, δ 10⁻⁵) | Exact ε, add/remove | ε if read as replace-one (RDP / exact) |
| ---: | ---: | ---: | ---: |
| 8 | 3.09 | 6.76 | 18.1 / 15.9 |
| 4 | 5.80 | 3.25 | 8.6 / 7.3 |
| 1 | 21.92 | 0.74 | 2.0 / 1.6 |

- The planner (`plan_central_dp`) says "add/remove adjacency" in its assumptions string. The
  `dp_mechanism` module docstring says add/remove. The **eval card and UI** only say "client-level
  (ε, δ)", which a reader will take as the stronger guarantee.
- **Direct framework calls are the weak spot.** If `dp_aggregate` is called on a cohort whose size
  varies with the data (for example the framework's q < 1 path, which the live server refuses), the
  data-dependent denominator makes even the add/remove sensitivity about 2S/n. The accounted ε is then
  understated by about a factor of two in z. That path also uses Poisson-subsampling RDP for what is
  actually fixed-size sampling, which is a second mismatch.

## Fix options (user decision)

1. **State the relation and fix the denominator (recommended).**
   - Record `adjacency: add/remove (fixed public denominator)` on the eval card and in the API.
   - Make `dp_aggregate` take an explicit public denominator W, and refuse if n ≠ W. That closes the
     data-dependent case.
   - No change to noise or utility on the live path.
2. **Switch to replace-one.** Double the noise to 2z·S/n so that the reported ε means replace-one,
   matching the capstone simulator. This costs utility. At the same claimed ε it roughly matches
   moving from ε 8 to about ε 3.3 under add/remove.
3. **Offer both.** Use option 1 by default, with a `--dp-adjacency replace-one` flag that doubles the
   noise and labels the card accordingly.

Whichever option is chosen, the capstone's simulator numbers are unaffected: they use replace-one
throughout, and their ε is the stronger claim.
