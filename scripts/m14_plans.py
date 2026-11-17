"""Milestone 14 run plans, generated from the frozen configuration (docs/M14_EXPERIMENTAL_PROTOCOL.md).

    python scripts/m14_plans.py voc        # DEV-A trigger validation (needs configs/m14_lambda.json)
    python scripts/m14_plans.py pilot      # DEV-A pilot comparison (needs configs/m14_frozen.json)
    python scripts/m14_plans.py confirm    # GEN2 / DEV-B / GEN, TN_s0-s2 (frozen)
    python scripts/m14_plans.py h5         # TNlp / TNse on GEN2
    python scripts/m14_plans.py ablate     # DEV-A + DEV-B ablations, TN_s0
    python scripts/m14_plans.py shift      # terrain shifts (DEV-B@x2, @rough; GEN2@x2)
    python scripts/m14_plans.py dm         # density-matched GEN2 runs (needs configs/m14_density.json)
    python scripts/m14_plans.py transfer   # lambda moment check on train_cal with TN_s1 / TN_s2

Every plan is written to configs/m14_plans/<name>.json; specs are ordered by priority (protocol §10).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urbanmorph.config import PROJECT_ROOT

CFG = PROJECT_ROOT / "configs"
LAM = CFG / "m14_lambda.json"
FROZEN = CFG / "m14_frozen.json"
SEEDS = [1, 2, 3]


def lam(mult=1.0, dead=True, island=True):
    L = json.loads(LAM.read_text())
    return {"lam_dead": round(L["lam_dead"] * mult, 4) if dead else 0.0,
            "lam_island": round(L["lam_island"] * mult, 4) if island else 0.0}


def methods(fz, include_slow=True):
    """(label, preset, overrides, extra spec keys) of the confirmatory method set."""
    out = [("A", "standard", {}, {}), ("M*", "energy", {"K": 8, "lam_ratio": 0.5}, {}),
           ("M*+repair", "energy", {"K": 8, "lam_ratio": 0.5}, {"repair": True}),
           ("P", "P", {}, {}),
           ("NA-myopic", "na_myopic", lam(), {}),
           ("NA-adaptive", "na_adaptive", {**lam(), "trigger": fz["trigger"]}, {}),
           ("NA-random", "na_adaptive", {**lam(), "trigger": f"random:{fz['fire_share']:.4f}"}, {})]
    if include_slow:
        out += [("NA-fixed", "na_fixed", lam(), {}),
                ("NA-SMC", "na_smc", lam(), {}),
                ("M13-SMC", "smc", {"K": 8, "lam_ratio": 0.5, "particles": 4}, {})]
    return out


def spec(tiles, bb, pre, ov, extra, seeds=SEEDS, potentials=None):
    s = {"tiles": tiles, "backbone": bb, "preset": pre, "overrides": ov, "seeds": seeds, **extra}
    if potentials:
        s["potentials"] = potentials
    return s


def plan(name):
    runs = []
    if name == "voc":
        runs = [spec("devA", "TN_s0", "na_fixed", lam(), {"voc": True}, [1, 2])]
    elif name == "transfer":
        runs = [spec("train_cal", b, "na_fixed", lam(), {}, [1, 2]) for b in ("TN_s1", "TN_s2")]
    else:
        fz = json.loads(FROZEN.read_text())
        if name == "pilot":
            runs = [spec("devA", "TN_s0", p, ov, ex, [1, 2]) for _, p, ov, ex in methods(fz)]
        elif name == "confirm":
            for tiles in ("gen2", "devB", "gen"):
                for _, p, ov, ex in methods(fz):
                    if p == "P":
                        runs.append(spec(tiles, "TN_s0", p, ov, ex))
                        continue
                    for b in ("TN_s0", "TN_s1", "TN_s2"):
                        runs.append(spec(tiles, b, p, ov, ex))
        elif name == "h5":
            for fam, pots in (("TNlp", "TNlp"), ("TNse", "TNse")):
                for b in (f"{fam}_s0", f"{fam}_s1", f"{fam}_s2"):
                    for _, p, ov, ex in methods(fz, include_slow=False):
                        if p in ("P",) or ex.get("repair"):
                            continue
                        runs.append(spec("gen2", b, p, ov, ex, [1, 2], potentials=pots))
        elif name == "ablate":
            for tiles in ("devA", "devB"):
                base = {**lam(), "trigger": fz["trigger"]}
                # calibration gave lam_island = 0, so "dead-end only" IS the frozen method; the island potential is
                # ablated at a fixed heuristic lam_island = 1 (labelled as such): island only, dead end + island
                variants = [("na_adaptive", {**base, "lam_dead": 0.0, "lam_island": 1.0}),
                            ("na_adaptive", {**base, "lam_island": 1.0}),
                            ("na_adaptive", {**base, "soft": False}),
                            ("na_adaptive", {**base, "horizon": 1}), ("na_adaptive", {**base, "horizon": 4}),
                            ("na_adaptive", {**base, "K": 4}), ("na_adaptive", {**base, "K": 16}),
                            ("na_myopic", {**lam(), "kl_eps": 0.05}), ("na_myopic", {**lam(), "kl_eps": 0.2})]
                for m in (0.5, 2.0, 4.0):
                    variants.append(("na_myopic", lam(m)))
                for f in ("island", "deg1", "water", "steep", "feas_low"):
                    variants.append(("na_adaptive", {**lam(), "trigger": f}))
                runs += [spec(tiles, "TN_s0", p, ov, {}, [1, 2]) for p, ov in variants]
        elif name == "shift":
            for tiles in ("devB@x2", "devB@rough", "gen2@x2"):
                for lab, p, ov, ex in methods(fz, include_slow=False):
                    if lab in ("A", "M*", "NA-myopic", "NA-adaptive"):
                        runs.append(spec(tiles, "TN_s0", p, ov, ex, [1, 2]))
        elif name == "dm":
            dm = json.loads((CFG / "m14_density.json").read_text())
            for lab, p, ov, ex in methods(fz):
                if lab in dm["stop_bias"]:
                    for b in ("TN_s0", "TN_s1", "TN_s2"):
                        runs.append(spec("gen2", b, p, {**ov, "stop_bias": dm["stop_bias"][lab]}, ex, [1, 2]))
        else:
            raise ValueError(name)
    out = CFG / "m14_plans" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": runs}, indent=1))
    print(name, len(runs), "runs ->", out)


if __name__ == "__main__":
    plan(sys.argv[1])
