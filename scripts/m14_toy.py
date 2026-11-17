"""Milestone 14 toy experiments: exact enumeration of the distributions in docs/M14_MATHEMATICAL_FORMULATION.md.

    python scripts/m14_toy.py      # -> outputs/m14/toy/{toy.json, toy.md, fig_toy.png}

A sequence x = (a_1..a_T), a_t in {0..A-1}, T = 8, A = 3 (6,561 sequences, all enumerated). Ingredients mirror the
road generator:
* prior p(a_t | a_{t-1}): a random Markov chain (Dirichlet(1) rows, one table per step; seeded);
* local "terrain" potential g_t(a_{t-1}, a_t) = exp(u), u ~ N(0, 0.5^2) (seeded);
* hard constraint: x must END with action 0 ("the road must reach the network"); a myopic sampler only sees it at T;
* network count potential Phi(x) = -#{t: (a_{t-1}, a_t) = (1, 2)} ("dead ends"), telescoping: Phi = sum_t phi_t.
Target pi_lam(x) ∝ p(x) 1[a_T = 0] prod_t g_t exp(lam phi_t).

Claims checked exactly (KL in nats):
C1 label bias: the locally normalised sampler q_loc(a_t | a_{t-1}) ∝ p g exp(lam phi) 1[feasible step] is not pi;
C2 twisting with the exact value function psi* gives q = pi; an H-step truncated twist approaches pi as H grows;
C3 finite-K self-normalised selection interpolates between the prior (K = 1) and q_loc (K -> inf);
C4 adaptive look-ahead: at equal budget (share of steps that use the H-step twist), a trigger beats random allocation;
C5 SMC with N particles and any positive twist converges to pi (Monte Carlo over 20,000 runs);
C6 I-projection: for a fixed E[Phi], pi_lam (the exponential tilt) has the smallest KL(q || p_F) among the samplers;
    p_F = the prior conditioned on feasibility.
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m14" / "toy"
T, A, SEED = 8, 3, 14
rng = np.random.default_rng(SEED)
P0 = rng.dirichlet(np.ones(A))                                    # p(a_1)
P = rng.dirichlet(np.ones(A), size=(T, A))                       # P[t, prev, a]
LOGG = rng.normal(0, 0.5, size=(T, A + 1, A))                    # local potential, row A = no previous action
XS = np.array(list(itertools.product(range(A), repeat=T)))


def phi_step(prev, a):
    return -1.0 if (prev == 1 and a == 2) else 0.0


def logp(x):
    return np.log(P0[x[0]]) + sum(np.log(P[t, x[t - 1], x[t]]) for t in range(1, T))


def logpot(x, lam):
    return LOGG[0, A, x[0]] + sum(LOGG[t, x[t - 1], x[t]] + lam * phi_step(x[t - 1], x[t]) for t in range(1, T))


LP = np.array([logp(x) for x in XS])
FEAS = XS[:, -1] == 0
PHI = np.array([sum(phi_step(x[t - 1], x[t]) for t in range(1, T)) for x in XS])


def normalise(logw):
    w = np.exp(logw - logw.max())
    return w / w.sum()


def target(lam):
    return normalise(np.where(FEAS, LP + np.array([logpot(x, lam) for x in XS]), -np.inf))


def kl(q, p):
    m = q > 0
    return float(np.sum(q[m] * (np.log(q[m]) - np.log(np.maximum(p[m], 1e-300)))))


# ---------------------------------------------------------------- step factors and exact / truncated value functions
def step_factor(t, prev, lam):
    """Unnormalised local factor f_t(prev, a) = p g exp(lam phi) 1[feasible step] for every a (vector)."""
    p = P0 if t == 0 else P[t, prev]
    lg = LOGG[t, A if t == 0 else prev]
    f = p * np.exp(lg + (0 if t == 0 else lam * np.array([phi_step(prev, a) for a in range(A)])))
    if t == T - 1:
        f = f * (np.arange(A) == 0)
    return f


def value(lam, H=None):
    """V[t][a] = sum over the next min(H, T-1-t) steps of products of local factors after choosing a at t
    (exact psi* when H >= T)."""
    V = [np.ones(A) for _ in range(T)]
    for t in range(T - 2, -1, -1):
        for a in range(A):
            if H is None:
                V[t][a] = float((step_factor(t + 1, a, lam) * V[t + 1]).sum())
    if H is None:
        return V
    Vh = [np.ones(A) for _ in range(T)]
    for t in range(T - 1):
        for a in range(A):
            # H-step truncated expectation: recurse H steps ahead, value 1 beyond the horizon
            def rec(s, prev, depth):
                if depth == 0 or s >= T:
                    return 1.0
                f = step_factor(s, prev, lam)
                return float(sum(f[b] * rec(s + 1, b, depth - 1) for b in range(A) if f[b] > 0))
            Vh[t][a] = rec(t + 1, a, H)
    return Vh


def sequential(lam, psi=None, use=None):
    """Distribution over XS of the single-chain sampler q(a_t | a_{t-1}) ∝ f_t(prev, a) psi_t(a)^[use(t, prev)]."""
    logq = np.zeros(len(XS))
    for i, x in enumerate(XS):
        lq, prev = 0.0, A
        for t in range(T):
            f = step_factor(t, prev if t else 0, lam) if t else step_factor(0, 0, lam)
            if psi is not None and (use is None or use(t, prev)):
                f = f * psi[t]
            if f[x[t]] <= 0:
                lq = -np.inf
                break
            lq += np.log(f[x[t]] / f.sum())
            prev = x[t]
        logq[i] = lq
    q = np.exp(logq)
    return q / q.sum()


def k_candidate(lam, K):
    """Exact single-chain K-candidate selection (candidates iid from the prior step, select ∝ g exp(lam phi) 1[ok])."""
    logq = np.zeros(len(XS))
    counts = [c for c in itertools.product(range(K + 1), repeat=A) if sum(c) == K]
    from math import factorial
    sel_cache = {}

    def select_probs(t, prev):
        key = (t, prev)
        if key in sel_cache:
            return sel_cache[key]
        p = P0 if t == 0 else P[t, prev]
        f = step_factor(t, prev, lam)
        w = np.where(p > 0, f / p, 0.0)                               # potential weight of each action
        out = np.zeros(A)
        for c in counts:
            pr = factorial(K) * np.prod([p[a] ** c[a] / factorial(c[a]) for a in range(A)])
            tot = sum(c[a] * w[a] for a in range(A))
            if tot <= 0:
                continue                                              # all candidates infeasible: sampler fails
            for a in range(A):
                out[a] += pr * c[a] * w[a] / tot
        sel_cache[key] = out
        return out
    for i, x in enumerate(XS):
        lq, prev = 0.0, 0
        for t in range(T):
            s = select_probs(t, prev)
            if s[x[t]] <= 0:
                lq = -np.inf
                break
            lq += np.log(s[x[t]])
            prev = x[t]
        logq[i] = lq
    q = np.exp(logq)
    return q / q.sum(), float(np.exp(logq).sum())                    # (conditional on success, success prob)


def smc(lam, N, psi=None, use=None, runs=20000, seed=0):
    """Empirical output distribution of SMC (multinomial resampling every step, one particle drawn at the end
    ∝ final weight) with the twisted proposal; weights = local normaliser x psi correction."""
    r = np.random.default_rng(seed)
    idx = {tuple(x): i for i, x in enumerate(XS)}
    hist = np.zeros(len(XS))
    for _ in range(runs):
        X = np.zeros((N, T), int)
        logw = np.zeros(N)
        prevpsi = np.ones(N)
        for t in range(T):
            for n in range(N):
                prev = X[n, t - 1] if t else 0
                f = step_factor(t, prev, lam)
                tw = psi[t] if (psi is not None and (use is None or use(t, prev if t else A))) else np.ones(A)
                g = f * tw
                Z = g.sum()
                if Z <= 0:
                    logw[n] = -np.inf
                    continue
                a = r.choice(A, p=g / Z)
                X[n, t] = a
                logw[n] += np.log(Z) - np.log(prevpsi[n])
                prevpsi[n] = tw[a]
            w = np.exp(logw - logw.max())
            w /= w.sum()
            if t < T - 1:
                anc = r.choice(N, size=N, p=w)
                X, prevpsi = X[anc], prevpsi[anc]
                logw = np.full(N, np.log(np.mean(np.exp(logw - logw.max()))) + logw.max())
        w = np.exp(logw - logw.max())
        k = r.choice(N, p=w / w.sum())
        hist[idx[tuple(X[k])]] += 1
    return hist / hist.sum()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lam = 2.0
    pi = target(lam)
    res = {"setup": {"T": T, "A": A, "seed": SEED, "lam": lam, "n_sequences": len(XS),
                     "pi_feasible_mass_under_prior": float(np.exp(LP)[FEAS].sum())}}
    # C1 / C2
    qloc = sequential(lam)
    Vstar = value(lam)
    qstar = sequential(lam, Vstar)
    res["C1_label_bias_KL(q_loc||pi)"] = kl(qloc, pi)
    res["C2_exact_twist_KL"] = kl(qstar, pi)
    res["C2_truncated"] = {H: kl(sequential(lam, value(lam, H)), pi) for H in (1, 2, 3, 4, 6, 8)}
    # C3
    res["C3_K_candidate"] = {}
    for K in (1, 2, 4, 8):
        qk, succ = k_candidate(lam, K)
        res["C3_K_candidate"][K] = {"KL(q_K||q_loc)": kl(qk, qloc), "KL(q_K||pi)": kl(qk, pi), "success": succ}
    # C4 adaptive allocation of a 2-step look-ahead
    H = 2
    VH = value(lam, H)
    q0 = qloc
    # oracle value of computation per (t, prev): TV between the step distributions with and without the twist
    voc = {}
    for t in range(T):
        for prev in range(A + 1):
            if (t == 0) != (prev == A):
                continue
            f = step_factor(t, prev if t else 0, lam)
            if f.sum() <= 0:
                continue
            g = f * VH[t]
            voc[(t, prev)] = 0.5 * float(np.abs(f / f.sum() - g / max(g.sum(), 1e-300)).sum())
    keys = list(voc)
    # cheap trigger without the oracle: "close to a constraint" = within H steps of the terminal constraint
    trig = lambda t, prev: t >= T - 1 - H
    budget = np.mean([trig(t, p) for t, p in keys])
    r = np.random.default_rng(1)
    rand_kl = []
    for _ in range(50):
        chosen = set(map(tuple, np.array(keys, dtype=object)[r.permutation(len(keys))[:int(round(budget * len(keys)))]]))
        rand_kl.append(kl(sequential(lam, VH, use=lambda t, p, c=chosen: (t, p) in c), pi))
    top = set(sorted(keys, key=lambda k: -voc[k])[:int(round(budget * len(keys)))])
    res["C4_adaptive"] = {"H": H, "budget_share": float(budget),
                          "KL_none": kl(q0, pi), "KL_fixed_all_steps": kl(sequential(lam, VH), pi),
                          "KL_trigger_near_constraint": kl(sequential(lam, VH, use=trig), pi),
                          "KL_random_same_budget_mean": float(np.mean(rand_kl)),
                          "KL_random_same_budget_p10_p90": [float(np.percentile(rand_kl, 10)), float(np.percentile(rand_kl, 90))],
                          "KL_oracle_voc_same_budget": kl(sequential(lam, VH, use=lambda t, p: (t, p) in top), pi)}
    # C5 SMC
    res["C5_smc"] = {}
    for N, lab, psi, use in ((8, "no twist", None, None), (8, "H=2 twist, adaptive", VH, trig), (8, "H=2 twist, all", VH, None),
                             (32, "no twist", None, None)):
        q = smc(lam, N, psi, use, runs=20000, seed=N)
        res["C5_smc"][f"N={N}, {lab}"] = kl(q, pi)
    res["C5_note"] = "Monte Carlo estimate of KL from 20,000 runs; the plug-in KL of an exact distribution sampled " \
                     "20,000 times is itself > 0 (reference row 'exact pi resampled')."
    emp = np.random.default_rng(5).multinomial(20000, pi) / 20000
    res["C5_smc"]["exact pi resampled (MC floor)"] = kl(emp, pi)
    # C6 frontier: E[Phi] vs KL(q || p_F)
    pF = normalise(np.where(FEAS, LP + np.array([logpot(x, 0.0) for x in XS]), -np.inf))
    lams = np.linspace(0, 4, 17)
    front = {"exact_tilt": [], "local": [], "K8": []}
    for l in lams:
        q = target(l)
        front["exact_tilt"].append((float((q * PHI).sum()), kl(q, pF)))
        q = sequential(l)
        front["local"].append((float((q * PHI).sum()), kl(q, pF)))
        q, _ = k_candidate(l, 8)
        front["K8"].append((float((q * PHI).sum()), kl(q, pF)))
    res["C6_frontier"] = {k: v for k, v in front.items()}
    (OUT / "toy.json").write_text(json.dumps(res, indent=1, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    Hs = list(res["C2_truncated"])
    ax[0].plot(Hs, [res["C2_truncated"][h] for h in Hs], "o-", color="#2a6f97", label="H-step twist (single chain)")
    ax[0].axhline(res["C1_label_bias_KL(q_loc||pi)"], color="#b23a48", ls="--", label="no twist (local)")
    ax[0].set_xlabel("look-ahead horizon H"), ax[0].set_ylabel("KL(q || π) [nats]"), ax[0].legend(frameon=False)
    ax[0].set_title("C1–C2: label bias shrinks with the horizon")
    c4 = res["C4_adaptive"]
    labs = ["none", "random\n(same budget)", "trigger", "oracle VoC\n(same budget)", "all steps"]
    vals = [c4["KL_none"], c4["KL_random_same_budget_mean"], c4["KL_trigger_near_constraint"], c4["KL_oracle_voc_same_budget"],
            c4["KL_fixed_all_steps"]]
    ax[1].bar(labs, vals, color=["#bbbbbb", "#bbbbbb", "#2a6f97", "#6c9a8b", "#555555"])
    ax[1].set_ylabel("KL(q || π)"), ax[1].set_title(f"C4: adaptive H={H} look-ahead, budget {budget:.0%} of steps")
    for k, col, m in (("exact_tilt", "#2a6f97", "o"), ("local", "#b23a48", "s"), ("K8", "#e09f3e", "^")):
        e, d = np.array(front[k]).T
        ax[2].plot(-e, d, m + "-", color=col, label={"exact_tilt": "exact tilt π_λ", "local": "local (K → ∞)",
                                                     "K8": "K = 8 candidates"}[k], ms=4)
    ax[2].set_xlabel("E[# dead-end events]  (lower = stronger network objective)"), ax[2].set_ylabel("KL(q || p_F)")
    ax[2].invert_xaxis(), ax[2].legend(frameon=False), ax[2].set_title("C6: deviation from the prior per unit of objective")
    fig.savefig(OUT / "fig_toy.png", dpi=150)
    md = ["# M14 toy experiments (exact enumeration)\n", "```", json.dumps({k: v for k, v in res.items() if k != "C6_frontier"},
                                                                         indent=1, default=float), "```"]
    (OUT / "toy.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "C6_frontier"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
