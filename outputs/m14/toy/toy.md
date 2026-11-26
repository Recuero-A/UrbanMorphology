# M14 toy experiments (exact enumeration)

```
{
 "setup": {
  "T": 8,
  "A": 3,
  "seed": 14,
  "lam": 2.0,
  "n_sequences": 6561,
  "pi_feasible_mass_under_prior": 0.06533970564535581
 },
 "C1_label_bias_KL(q_loc||pi)": 1.123115122202566,
 "C2_exact_twist_KL": 4.8935890303399676e-17,
 "C2_truncated": {
  "1": 0.24871500199733151,
  "2": 0.08176629018722405,
  "3": 0.013815762396242432,
  "4": 0.0010977786387739406,
  "6": 8.286626976879304e-07,
  "8": 4.8935890303399676e-17
 },
 "C3_K_candidate": {
  "1": {
   "KL(q_K||q_loc)": 0.9923665666873396,
   "KL(q_K||pi)": 0.6384440534368906,
   "success": 0.06533970564535581
  },
  "2": {
   "KL(q_K||q_loc)": 0.5348089089553096,
   "KL(q_K||pi)": 0.43150213214652205,
   "success": 0.11811206465009601
  },
  "4": {
   "KL(q_K||q_loc)": 0.36443580338083226,
   "KL(q_K||pi)": 0.41558008354315873,
   "success": 0.21476789893711357
  },
  "8": {
   "KL(q_K||q_loc)": 0.2611090034102845,
   "KL(q_K||pi)": 0.4511697024168218,
   "success": 0.36197252026076854
  }
 },
 "C4_adaptive": {
  "H": 2,
  "budget_share": 0.4090909090909091,
  "KL_none": 1.123115122202566,
  "KL_fixed_all_steps": 0.08176629018722405,
  "KL_trigger_near_constraint": 0.771000801928513,
  "KL_random_same_budget_mean": 0.6206420323489976,
  "KL_random_same_budget_p10_p90": [
   0.2881824206177847,
   0.9194407704107455
  ],
  "KL_oracle_voc_same_budget": 0.27252548519372494
 },
 "C5_smc": {
  "N=8, no twist": 0.11510354875585013,
  "N=8, H=2 twist, adaptive": 0.09704904941497063,
  "N=8, H=2 twist, all": 0.033505093972425654,
  "N=32, no twist": 0.028756612622931055,
  "exact pi resampled (MC floor)": 0.025369861479588827
 },
 "C5_note": "Monte Carlo estimate of KL from 20,000 runs; the plug-in KL of an exact distribution sampled 20,000 times is itself > 0 (reference row 'exact pi resampled')."
}
```
