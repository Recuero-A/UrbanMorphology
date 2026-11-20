"""Shared experiment infrastructure for the inference-time milestones (M13 - M15).

These helpers used to live in milestone scripts and were imported across scripts. They are library code:

* ``backbones``  load a trained TerrainGraphTurtle checkpoint; what terrain the MODEL sees (leakage controls);
* ``methods``    the single registry of named sampler configurations (A, M*, M13-SMC, NA-*, M15 style variants);
* ``graph_io``   RoadGraph <-> JSON (the format of every stored run);
* ``jsonio``     atomic JSON writes for resumable runs.

The milestone scripts (scripts/m13_*.py, scripts/m14_*.py) re-export these names, so historical commands still work.
"""
