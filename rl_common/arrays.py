"""Tiny helpers for moving backend arrays to the host."""

from __future__ import annotations

import numpy as np


def to_numpy(x) -> np.ndarray:
    """Host copy of a Warp array, a JAX array or anything numpy understands."""
    if hasattr(x, "numpy"):  # warp.array
        return x.numpy()
    return np.asarray(x)
