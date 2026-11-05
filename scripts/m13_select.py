"""Milestone 13 staged selection on DEV-A (docs/M13_EVALUATION_PROTOCOL.md §7), applied automatically.

    python scripts/m13_select.py      # -> outputs/m13/selection.json, outputs/m13/best_dev_result.md

Backbone TN_s0, inference seeds 1-2. Rule: eligible = mean street density within +-15 % of real AND premature share
<= that of A (standard) + 0.05; choose the lowest S_T; ties within 0.02 go to the faster configuration. If nothing is
eligible, the lowest S_T among configurations meeting the premature criterion is taken and flagged as a density-
criterion failure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import m13_analyze as an  # noqa: E402
import m13_experiments as ex  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m13"
BASE = {"tiles": "devA", "backbone": "TN_s0", "potentials": "TN", "seeds": [1, 2]}


def spec(preset, **ov):
    return {**BASE, "preset": preset, **({"overrides": ov} if ov else {})}


def evaluate(specs, device):
    out = []
    for s in specs:
        rid = ex.run(s, device)
        res = json.loads((ex.RUNS / f"{rid}.json").read_text())
        out.append(an.summarize(res))
    return out


def choose(summaries, prem_a):
    ok = [s for s in summaries if abs(s["density_ratio"] - 1) <= 0.15 and s["premature_share"] <= prem_a + 0.05]
    flag = ""
    if not ok:
        ok = [s for s in summaries if s["premature_share"] <= prem_a + 0.05]
        flag = "density criterion failed for every configuration"
    if not ok:
        return None, "no configuration meets the premature criterion"
    best = min(s["S_T"] for s in ok)
    tie = [s for s in ok if s["S_T"] <= best + 0.02]
    return min(tie, key=lambda s: s["sec_per_tile_sample"]), flag


def ov_of(s):
    return dict(s["spec"].get("overrides", {})), s["spec"]["preset"]


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log = {"rule": __doc__.split("Rule: ")[1].strip(), "stages": {}}
    # ---- S1 one-step
    s1 = [spec("standard"), spec("legacy")]
    s1 += [spec("topk", K=8, topk=k) for k in (1, 2, 4)]
    s1 += [spec("fulltemp", full_temperature=t) for t in (0.5, 0.7, 1.0)]
    s1 += [spec("filter", K=k) for k in (4, 8, 16, 32)]
    s1 += [spec("energy", K=k, lam_ratio=l) for k in (4, 8, 16, 32) for l in (0.5, 1.0, 2.0)]
    s1 += [spec("softonly", K=8, lam_ratio=1.0)]
    r1 = evaluate(s1, device)
    prem_a = next(s for s in r1 if s["spec"]["preset"] == "standard")["premature_share"]
    log["premature_A"] = prem_a
    w1, f1 = choose(r1, prem_a)
    e_star, fe = choose([s for s in r1 if s["spec"]["preset"] == "energy"], prem_a)
    log["stages"]["S1"] = {"winner": w1["id"], "flag": f1, "E_star": e_star["id"], "E_flag": fe}
    # ---- S2 look-ahead on E*
    ov, _ = ov_of(e_star)
    r2 = evaluate([spec("energy_la", **{**ov, "horizon": h, "rollout_m": 4}) for h in (1, 2, 4, 8)], device)
    h_best, fh = choose(r2, prem_a)
    log["stages"]["S2"] = {"winner": h_best["id"] if h_best else None, "flag": fh,
                           "H_star": (h_best or {}).get("spec", {}).get("overrides", {}).get("horizon")}
    pool = r1 + r2
    w12, _ = choose(pool, prem_a)
    # ---- S3 particles on the best of S1 + S2
    ov, pre = ov_of(w12)
    H = ov.get("horizon", 0)
    base_ov = {k: v for k, v in ov.items() if k in ("K", "lam_ratio")}
    s3 = [spec("smc", **{**base_ov, "particles": n, **({"horizon": H, "rollout_m": 4} if H else {})}) for n in (4, 8)]
    s3 += [spec("beam", **{**base_ov, "particles": 4})]
    hs = log["stages"]["S2"]["H_star"] or 2
    s3 += [spec("smc_la", **{**base_ov, "particles": 4, "horizon": hs, "rollout_m": 4})]
    r3 = evaluate(s3, device)
    pool += r3
    w123, f3 = choose(pool, prem_a)
    log["stages"]["S3"] = {"winner": w123["id"], "flag": f3}
    # ---- S4 scheduler on the best so far
    ov, pre = ov_of(w123)
    r4 = evaluate([{**BASE, "preset": pre, "overrides": {**ov, "scheduler": sc}} for sc in
                   ("fifo", "class", "confidence", "connect", "stochastic")], device)
    pool += r4
    m_star, f4 = choose(pool, prem_a)
    log["stages"]["S4"] = {"winner": m_star["id"], "flag": f4}
    log["M_star"] = {"id": m_star["id"], "preset": m_star["spec"]["preset"], "overrides": m_star["spec"].get("overrides", {})}
    log["E_star"] = {"id": e_star["id"], "preset": "energy", "overrides": e_star["spec"].get("overrides", {})}
    log["H_star"] = log["stages"]["S2"]["H_star"]
    log["summaries"] = {s["id"]: {k: v for k, v in s.items() if k != "spec"} | {"spec": s["spec"]} for s in pool}
    (OUT / "selection.json").write_text(json.dumps(log, indent=1, default=float))
    b = log["summaries"][m_star["id"]]
    (OUT / "best_dev_result.md").write_text(
        f"Selected M* (DEV-A, TN_s0): `{m_star['spec']['preset']}` {m_star['spec'].get('overrides', {})}: S_T {b['S_T']:.3f}, "
        f"density {b['density']:.1f} vs real {b['density_real']:.1f} km/km2, premature {b['premature_share']:.2f}, "
        f"giant share {b['giant_share']:.2f}, S10_cond {b['S10_cond']:.3f}. Selection only; confirmation on DEV-B pending.\n")
    print(json.dumps({k: log[k] for k in ("M_star", "E_star", "H_star")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
