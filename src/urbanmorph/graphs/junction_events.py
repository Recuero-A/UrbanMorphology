"""Milestone 17 diagnosis: an event log of committed graph edits (SamplerConfig.event_log; off by default).

For every committed edit (TCSampler.run, any sampler built on it, NA and SMC included) one record is kept in the
generating state (``GenState._events``; SMC copies carry their own logs, the chosen particle's log is returned):
* ``kind``   the insertion rule that resolved the move: new_node, merge_node (step ends within 8 m of a node),
             crossing_split (cuts an existing street), crossing_endpoint, t_junction (ends within 8 m of a street),
             struct_new / struct_merge (bridge or tunnel);
* ``st``, ``cls``, ``src`` (model / style), ``lw`` (log weight), ``lw_net`` (M14 network term; > 0 = a closure reward);
* ``births``  the points that become junctions (degree 3) through this edit, each with the distance to the nearest
             junction that existed before the edit and its role: ``split`` (the new node of a crossing split or T),
             ``v`` (an existing node reached by a merge), ``u`` (the expanding node gaining a branch);
* ``reuse``  when a split / v birth lands closer than ``R_DIAG`` (15 m) to an existing junction j: whether the move
             u -> j, resolved by the same insertion rules on the same state, ends exactly at j and passes the hard
             rules (the topology-preserving alternative; None when not applicable);
* ``n_feasible`` / ``n_close`` feasible candidates of the decision and how many would create a close birth;
  ``avoidable`` = the chosen edit creates a close birth while a feasible candidate creating none existed.
After the edit, ``born`` lists the coordinates of nodes whose degree reached 3, so a close pair of the final graph
can be traced to the edit at which its later junction appeared.
"""
from __future__ import annotations

import numpy as np

R_DIAG = 15.0
CLOSING = ("merge_node", "crossing_endpoint", "struct_merge")
SPLITS = ("crossing_split", "t_junction")


def _junctions(s):
    J = np.array([k for k in range(s.n) if len(s.adj[k]) >= 3], int)
    return J, (s.nodes[J] if len(J) else np.zeros((0, 2)))


def _births(s, u, c):
    o = c["o"]
    k = o["kind"]
    out = []
    if k in SPLITS:
        out.append(("split", np.asarray(o["q"], float)))
    elif k in CLOSING and o.get("v", -1) >= 0 and len(s.adj[o["v"]]) == 2:
        out.append(("v", s.nodes[o["v"]].copy()))
    if len(s.adj[u]) == 2:
        out.append(("u", s.nodes[u].copy()))
    return out


def pre(sampler, i, s, u, cs, j) -> dict:
    J, P = _junctions(s)

    def dnn(x):
        return float(np.min(np.hypot(*(P - x).T))) if len(P) else np.inf

    feas = [k for k, c in enumerate(cs) if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"]
    close = {k: any(dnn(x) < R_DIAG for _, x in _births(s, u, cs[k])) for k in feas}
    c = cs[j]
    births = [(w, x, dnn(x)) for w, x in _births(s, u, c)]
    reuse = None
    for w, x, d in births:
        if w in ("split", "v") and d < R_DIAG:
            jn = int(J[np.argmin(np.hypot(*(P - x).T))])
            dd = s.nodes[jn] - s.nodes[u]
            if np.hypot(*dd) < 1.0 or jn in s.adj[u]:
                reuse = False
                break
            o2 = s.resolve_step(u, dd)
            ok = o2["kind"] in CLOSING and int(o2.get("v", -1)) == jn
            if ok:
                _, _, lw2, reason = sampler.evaluate_candidate(i, s, u, dd, int(c["cls"]), 0)
                ok = bool(np.isfinite(lw2) and not reason)
            reuse = bool(ok)
            break
    chosen_close = any(d < R_DIAG for _, _, d in births)
    return {"kind": c["o"]["kind"], "st": int(c["st"]), "cls": int(c["cls"]), "src": c.get("src", "model"),
            "lw": float(c["lw"]) if np.isfinite(c["lw"]) else None,
            "lw_net": float(c.get("lw_net", 0.0)), "births": [[w, float(x[0]), float(x[1]), d] for w, x, d in births],
            "reuse": reuse, "n_feasible": len(feas), "n_close": int(sum(close.values())),
            "chosen_close": bool(chosen_close),
            "avoidable": bool(chosen_close and any(not v for v in close.values())),
            "u": [float(s.nodes[u][0]), float(s.nodes[u][1])], "born": []}


def post(s, u, v, ev) -> None:
    jb = s.__dict__.setdefault("_jbirth", {})
    evs = s.__dict__.setdefault("_events", [])
    for node in (u, v):
        if node is not None and node >= 0 and len(s.adj[node]) >= 3 and node not in jb:
            jb[node] = len(evs)
            ev["born"].append([float(s.nodes[node][0]), float(s.nodes[node][1])])
    evs.append(ev)
