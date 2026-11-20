"""Atomic JSON writes: a run file is either the old or the new version, never half written."""
from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    os.replace(tmp, path)
