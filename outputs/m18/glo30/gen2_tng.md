# M18: the GLO-30 backbone TNg vs TN on GEN2 (95 tiles, 3 backbones × 3 seeds)

Real GEN2: OD unreachability 0.047, 13.9 km/km². S_T and S10: W1/IQR gap to the real networks (lower = closer). Script `scripts/m18_tng_gen2.py`.

| Backbone | Method | OD unreach. | islands / tile | km/km² | S_T | S10 |
|---|---|---|---|---|---|---|
| TN | M* | 0.036 | 0.28 | 15.5 | 0.189 | 0.345 |
| TN | M*+repair | 0.006 | 0.02 | 16.0 | 0.381 | 0.406 |
| TN | NA-SMC | 0.011 | 0.02 | 15.5 | 0.202 | 0.323 |
| TNg | M* | 0.023 | 0.18 | 13.8 | 0.458 | 0.317 |
| TNg | M*+repair | 0.004 | 0.01 | 14.1 | 0.552 | 0.336 |
| TNg | NA-SMC | 0.013 | 0.01 | 12.1 | 0.356 | 0.380 |

Paired block contrast NA-SMC − M\* on OD unreachability (m14_analyze.block_contrast):

* TN: Δ -0.0247 [-0.0379, -0.0099], sign-flip p 0.00145, per backbone -0.0182, -0.0273, -0.0287
* TNg: Δ -0.0093 [-0.0207, +0.0087], sign-flip p 0.3086, per backbone -0.0129, -0.0034, -0.0117
