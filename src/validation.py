from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PurgedSplitConfig:
    """Configuration for leakage-aware time-series validation.

    ``label_horizon`` and ``embargo`` are measured in rows of the validation
    frame.  For weekly SOFR labels based on a 63 business-day forward realized
    window, a 13-row horizon is the natural default.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    label_horizon: int = 13
    embargo: int = 2


def _contiguous_blocks(n_obs: int, n_groups: int) -> list[np.ndarray]:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2.")
    if n_groups > n_obs:
        raise ValueError("n_groups cannot exceed the number of observations.")
    return [block.astype(int) for block in np.array_split(np.arange(n_obs), n_groups) if len(block)]


def _split_contiguous_indices(indices: np.ndarray) -> list[np.ndarray]:
    indices = np.sort(np.asarray(indices, dtype=int))
    if len(indices) == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    return [block.astype(int) for block in np.split(indices, split_points)]


def _purged_train_indices(
    n_obs: int,
    test_indices: np.ndarray,
    label_horizon: int,
    embargo: int,
) -> np.ndarray:
    """Return train rows whose label windows do not overlap the test region."""

    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative.")
    test_indices = np.asarray(test_indices, dtype=int)
    if len(test_indices) == 0:
        raise ValueError("test_indices cannot be empty.")

    starts = np.arange(n_obs)
    ends = starts + int(label_horizon)
    keep = np.ones(n_obs, dtype=bool)
    keep[test_indices] = False

    # Purge against every contiguous test block.  This supports CPCV splits
    # where test groups are not adjacent.
    for block in _split_contiguous_indices(test_indices):
        block_start = max(0, int(block.min()) - int(embargo))
        block_end = min(n_obs - 1, int(block.max()) + int(embargo))
        overlaps = (starts <= block_end) & (ends >= block_start)
        keep &= ~overlaps

    return np.flatnonzero(keep)


def purged_blocked_splits(
    index: Iterable[object],
    n_splits: int = 5,
    label_horizon: int = 13,
    embargo: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological blocked folds with purge and embargo controls."""

    n_obs = len(pd.Index(index))
    blocks = _contiguous_blocks(n_obs, n_splits)
    splits = []
    for test in blocks:
        train = _purged_train_indices(n_obs, test, label_horizon, embargo)
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def combinatorial_purged_splits(
    index: Iterable[object],
    config: PurgedSplitConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create CPCV-style train/test splits from combinations of time blocks."""

    cfg = config or PurgedSplitConfig()
    n_obs = len(pd.Index(index))
    blocks = _contiguous_blocks(n_obs, cfg.n_groups)
    if cfg.n_test_groups < 1 or cfg.n_test_groups >= len(blocks):
        raise ValueError("n_test_groups must be between 1 and n_groups - 1.")

    splits = []
    for group_ids in combinations(range(len(blocks)), cfg.n_test_groups):
        test = np.sort(np.concatenate([blocks[i] for i in group_ids])).astype(int)
        train = _purged_train_indices(n_obs, test, cfg.label_horizon, cfg.embargo)
        if len(train) and len(test):
            splits.append((train, test))
    return splits
