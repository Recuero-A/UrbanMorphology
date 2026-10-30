"""Saving, loading and validating processed layout samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from urbanmorph.config import CHANNELS, PROCESSED_DIR

EXPECTED_SHAPE = (3, 128, 128)
EXPECTED_DTYPE = np.uint8

REQUIRED_METADATA_KEYS = (
    "sample_id",
    "source",
    "attribution",
    "geographic_bounds",
    "projected_crs",
    "projected_bounds_m",
    "physical_size_m",
    "resolution_px",
    "pixel_size_m",
    "channels",
    "road_buffer",
    "processing_date_utc",
)


class LayoutValidationError(ValueError):
    pass


def sample_paths(sample_id: str, directory: Path = PROCESSED_DIR) -> dict[str, Path]:
    return {
        "array": directory / f"{sample_id}.npy",
        "metadata": directory / f"{sample_id}.json",
        "vectors": directory / f"{sample_id}_vectors.gpkg",
    }


def validate_layout(arr: np.ndarray, expected_shape: tuple[int, ...] = EXPECTED_SHAPE) -> None:
    if arr.shape != expected_shape:
        raise LayoutValidationError(f"shape {arr.shape} != expected {expected_shape}")
    if arr.dtype != EXPECTED_DTYPE:
        raise LayoutValidationError(f"dtype {arr.dtype} != expected {EXPECTED_DTYPE}")
    unique = np.unique(arr)
    if not np.isin(unique, (0, 1)).all():
        raise LayoutValidationError(f"non-binary values present: {unique[~np.isin(unique, (0, 1))]}")


def validate_metadata(meta: dict, arr: np.ndarray | None = None) -> None:
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise LayoutValidationError(f"metadata missing keys: {missing}")
    if {int(k): v for k, v in meta["channels"].items()} != CHANNELS:
        raise LayoutValidationError(f"channel definition mismatch: {meta['channels']}")
    b = meta["projected_bounds_m"]
    size, res = meta["physical_size_m"], meta["resolution_px"]
    if not (np.isclose(b["xmax"] - b["xmin"], size) and np.isclose(b["ymax"] - b["ymin"], size)):
        raise LayoutValidationError("projected bounds do not match physical_size_m")
    if not np.isclose(size / res, meta["pixel_size_m"]):
        raise LayoutValidationError("pixel_size_m != physical_size_m / resolution_px")
    if arr is not None and arr.shape[1:] != (res, res):
        raise LayoutValidationError("array resolution does not match metadata")


def save_sample(arr: np.ndarray, metadata: dict, sample_id: str, directory: Path = PROCESSED_DIR) -> dict[str, Path]:
    validate_layout(arr)
    validate_metadata(metadata, arr)
    paths = sample_paths(sample_id, directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(paths["array"], arr)
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return paths


def load_sample(sample_id: str, directory: Path = PROCESSED_DIR) -> tuple[np.ndarray, dict]:
    """Load a layout array and its metadata, validating both."""
    paths = sample_paths(sample_id, directory)
    arr = np.load(paths["array"])
    validate_layout(arr)
    meta = json.loads(paths["metadata"].read_text())
    validate_metadata(meta, arr)
    return arr, meta
