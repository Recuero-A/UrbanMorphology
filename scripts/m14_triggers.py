"""Milestone 14 trigger selection and validation on DEV-A (docs/M14_EXPERIMENTAL_PROTOCOL.md §4).

    python scripts/m14_triggers.py select     # VoC logs of the voc plan -> configs/m14_frozen.json, outputs/m14/triggers/
    python scripts/m14_triggers.py validate   # after the DEV-A pilot: adaptive vs random on od_unreach_land (recorded)

The selection rule is fixed in the protocol: among candidate triggers capturing >= 70 % of the total VoC, the one with
the smallest firing share (else the best VoC-per-share, flagged). Validation results are appended to the frozen file;
they never change the chosen trigger.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m14_analyze as ma  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m14" / "triggers"
FROZEN = PROJECT_ROOT / "configs" / "m14_frozen.json"


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted((PROJECT_ROOT / "outputs" / "m14" / "voc").glob("devA__TN_s0__na_fixed__*__s*.json"))
    lam = json.loads((PROJECT_ROOT / "configs" / "m14_lambda.json").read_text())
    want = {"lam_dead": round(lam["lam_dead"], 4), "lam_island": round(lam["lam_island"], 4)}
    used = []
    for f in files:
        rid = f.name.split("__s")[0]
        spec = json.loads((ma.RUNS / f"{rid}.json").read_text())["spec"]
        if spec["overrides"] == want and spec.get("voc"):
            rows += json.loads(f.read_text())
            used.append(f.name)
    table, total, n = ma.trigger_table(rows)
    best, flagged = ma.choose_trigger(table)
    voc = np.array([r["voc"] for r in rows])
    frozen = {"trigger": best["trigger"], "fire_share": best["fire_share"], "voc_captured": best["voc_captured"],
              "flagged_below_70pct": flagged, "random_expected_capture": best["fire_share"],
              "beats_random_in_expectation": best["voc_captured"] > best["fire_share"],
              "n_decisions": n, "total_voc": total, "mean_voc": float(voc.mean()),
              "voc_share_zero": float((voc < 1e-6).mean()), "source": used, "lambda": want,
              "rule": "protocol §4: smallest firing share among triggers capturing >= 70 % of total VoC (DEV-A, TN_s0)"}
    FROZEN.write_text(json.dumps(frozen, indent=1))
    (OUT / "trigger_table.json").write_text(json.dumps(table, indent=1))
    md = ["| trigger | fires on | VoC captured | captured / fired |", "|---|---|---|---|"]
    md += [f"| {r['trigger']} | {r['fire_share']:.1%} | {r['voc_captured']:.1%} | {r['ratio']:.2f} |" for r in table[:25]]
    (OUT / "trigger_table.md").write_text("\n".join(md) + "\n")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    xs = [r["fire_share"] for r in table]
    ys = [r["voc_captured"] for r in table]
    ax.scatter(xs, ys, s=14, color="#8a8a8a", label="candidate triggers")
    ax.plot([0, 1], [0, 1], "--", color="#b23a48", lw=1, label="random allocation (expected)")
    ax.scatter([best["fire_share"]], [best["voc_captured"]], s=60, color="#2a6f97", zorder=3,
               label=f"selected: {best['trigger']}")
    ax.axhline(0.7, color="#cccccc", lw=0.8)
    ax.set_xlabel("share of decisions with roll-outs"), ax.set_ylabel("share of total value of computation captured")
    ax.legend(frameon=False, fontsize=8), ax.set_title("Trigger validation on DEV-A (TN_s0, seeds 1–2)")
    fig.savefig(OUT / "fig_trigger_voc.png", dpi=150)
    print(json.dumps(frozen, indent=1))


def validate():
    fz = json.loads(FROZEN.read_text())
    lam = fz["lambda"]
    ad = ma.find("devA", "TN_s0", "na_adaptive", {**lam, "trigger": fz["trigger"]})
    rd = ma.find("devA", "TN_s0", "na_adaptive", {**lam, "trigger": f"random:{fz['fire_share']:.4f}"})
    fx = ma.find("devA", "TN_s0", "na_fixed", lam)
    sa, sr, sf = ma.summary(ad), ma.summary(rd), ma.summary(fx)
    val = {k: {"adaptive": sa[k], "random": sr[k], "fixed": sf[k]} for k in
           ("od_unreach_land", "dead_end_ratio_m13", "islands", "S10_cond", "density_ratio", "sec_per_tile_sample",
            "la_fire_share")}
    val["adaptive_le_random_on_od_unreach_land"] = sa["od_unreach_land"] <= sr["od_unreach_land"]
    val["justified"] = bool(fz["beats_random_in_expectation"] and val["adaptive_le_random_on_od_unreach_land"])
    fz["validation_devA"] = val
    FROZEN.write_text(json.dumps(fz, indent=1, default=float))
    print(json.dumps(val, indent=1, default=float))


if __name__ == "__main__":
    {"select": select, "validate": validate}[sys.argv[1]]()
