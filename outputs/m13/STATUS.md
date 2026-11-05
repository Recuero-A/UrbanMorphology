# Milestone 13 runner status

Updated 2026-11-05T17:08:32Z (UTC). Commit 16abf9a.

Active compute: 1.83 h of the 36 h budget.
Jobs: 2 of 18 completed.
Current: **train_TNse_s2** (since 2026-11-05T16:59:31Z).
Last checkpoint: checkpoints/m12/TNse_s2/latest.pt (02:08:29)
Estimated remaining: 24.4 h (from per-job estimates).

| Job | Status | Attempts | Start | End | Exit |
|---|---|---|---|---|---|
| train_TNse_s0 | completed | 2 | 2026-11-05T15:37:14Z | 2026-11-05T15:58:52Z | 0 |
| train_TNse_s1 | completed | 1 | 2026-11-05T15:58:52Z | 2026-11-05T16:39:32Z | 0 |
| train_TNse_s2 | running | 2 | 2026-11-05T16:59:31Z | 2026-11-05T16:58:44Z | 0 |
| train_TN_s2 | pending | 0 |  |  |  |
| train_TNlp_s2 | pending | 0 |  |  |  |
| train_N0_s2 | pending | 0 |  |  |  |
| calibrate_TNse | pending | 0 |  |  |  |
| select | pending | 0 |  |  |  |
| confirm_tn | pending | 0 |  |  |  |
| confirm_leak | pending | 0 |  |  |  |
| confirm_dm | pending | 0 |  |  |  |
| gen_data | pending | 0 |  |  |  |
| confirm_gen | pending | 0 |  |  |  |
| confirm_ablate | pending | 0 |  |  |  |
| x5_global | pending | 0 |  |  |  |
| trace | pending | 0 |  |  |  |
| report | pending | 0 |  |  |  |
| demos | pending | 0 |  |  |  |

## Current best development result

Selected M* (DEV-A, TN_s0): `energy` {'K': 8, 'lam_ratio': 0.5}: S_T 0.338, density 12.1 vs real 13.0 km/km2, premature 0.12, giant share 0.77, S10_cond 0.392. Selection only; confirmation on DEV-B pending.

