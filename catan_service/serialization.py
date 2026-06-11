"""Shared serialization helpers for service contracts and NumPy-backed objects."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively convert service payloads to JSON-safe native Python objects."""

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.int8, np.int16, np.int32, np.int64, np.integer)):
        return int(value)
    if isinstance(value, (np.float16, np.float32, np.float64, np.floating)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
