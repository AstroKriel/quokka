# One finding, before and after

Same concern, same evidence, from a real review.

Bad, 118 words:

> **1. `num_periods` has no input validation; non-positive values make the test pass vacuously.** `setup.num_periods=0` and `setup.num_periods=-1` both produce `stopTime_ <= 0`, run zero timesteps, and exit 0 with "error norm 6.368370e-18 (tol = 2.000000e-03)" — a green PASS for a simulation that never advanced. A negative value is worse than a no-op: it turns a correctness test into an unconditional success. These same files already validate their other setup parameters (`testFieldLoop.cpp:234` aborts on `loop_radius <= 0`), so guarding is the established convention here and the new parameter is the odd one out. Suggest an `amrex::Abort` in each of the seven sites.

Good, 34 words:

> 1. **`num_periods` unvalidated** — all 7 sites. `setup.num_periods=0` or `=-1` runs zero timesteps and reports PASS at error norm 6.37e-18; these files already abort on bad `loop_radius` (`testFieldLoop.cpp:234`).

Same claim, same `file:line`, same number, same fix — implied by "unvalidated". A five-finding review written this way lands near **165 words**, so the 300 cap binds only on genuinely large reviews.
