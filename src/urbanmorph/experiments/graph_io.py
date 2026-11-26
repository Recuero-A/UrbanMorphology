"""RoadGraph <-> JSON, the graph format of every stored run (outputs/m1{3,4,5}/runs/*.json)."""
from __future__ import annotations

import numpy as np

from urbanmorph.graphs.road_graph import RoadGraph


def graph_to_json(g: RoadGraph) -> dict:
    """Node coordinates rounded to 0.01 m; ``cls`` road class 0 major .. 3 minor; ``st`` 0 ground, 1 bridge, 2 tunnel."""
    return {"nodes": g.nodes.round(2).tolist(), "edges": g.edges.tolist(),
            "cls": np.asarray(g.classes()).tolist(), "st": np.asarray(g.structs()).tolist()}


def graph_from_json(G: dict) -> RoadGraph:
    nodes = np.array(G["nodes"], float).reshape(-1, 2)
    return RoadGraph(nodes, np.array(G["edges"], int).reshape(-1, 2), np.zeros(len(nodes), bool),
                     edge_class=np.array(G["cls"], int), edge_struct=np.array(G["st"], int))
