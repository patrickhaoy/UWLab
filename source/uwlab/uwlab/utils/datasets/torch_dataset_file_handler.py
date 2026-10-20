# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Torch Dataset File Handler
This module provides a dataset file handler that saves data directly in the torch format.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import torch
from collections.abc import Iterable
from typing import Any

from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase, EpisodeData


class TorchDatasetFileHandler(DatasetFileHandlerBase):
    """
    Dataset file handler that saves data directly in torch format as a torch file.
    """

    def __init__(self, flush_interval_s: float = 0.0):
        """Limit checkpoint frequency when requested; zero preserves immediate saves.

        The first flush and close always save pending data. Between checkpoints,
        newly collected states remain in memory.
        """
        if not math.isfinite(flush_interval_s) or flush_interval_s < 0:
            raise ValueError("flush_interval_s must be finite and non-negative")
        self._flush_interval_s = flush_interval_s
        self._last_flush_time = -math.inf
        self._dirty = False
        self._file_path: str | None = None
        self._episode_data: dict[str, Any] = {}
        self._episode_count: int = 0

    def create(self, file_path: str, env_name: str | None = None):
        """Create a new dataset file."""
        self._file_path = file_path
        self._episode_data = {}
        self._episode_count = 0
        self._last_flush_time = -math.inf
        self._dirty = False

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        self._last_flush_time = -math.inf
        self._dirty = False
        if mode == "r":
            if os.path.exists(file_path):
                self._episode_data = torch.load(file_path)
                self._file_path = file_path
            else:
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
        else:
            self._file_path = file_path
            self._episode_data = {}

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        return []

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = True):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
            dataset_compression: Accepted for interface compatibility and ignored.
                Isaac Lab 3.0's RecorderManager passes ``cfg.dataset_compression``
                positionally to every handler (see
                :meth:`isaaclab.managers.RecorderManager.export_episodes`); omitting
                it raises ``TypeError: write_episode() takes from 2 to 3 positional
                arguments but 4 were given`` at the first episode export. It is
                meaningful only for the HDF5 handler, which maps it onto gzip
                per-dataset compression. This handler accumulates episodes in memory
                and serializes them with ``torch.save`` in :meth:`flush`, which has no
                equivalent option.
        """
        if episode.is_empty() or not episode.success:
            return

        # Extract only the last entry from this episode using extend_dicts_last_entry logic
        self._extend_dicts_last_entry(self._episode_data, episode.data)
        self._dirty = True

        # Only increment episode count if using default indexing
        if demo_id is None:
            self._episode_count += 1

    def _extend_dicts_last_entry(self, dest: dict[str, Any], src: dict[str, Any], store_data: bool = True) -> None:
        """Extend destination dictionary with last entry from source dictionary."""
        for key in src.keys():
            if isinstance(src[key], dict):
                if key not in dest:
                    dest[key] = {}
                self._extend_dicts_last_entry(dest[key], src[key], store_data)
            else:
                if key not in dest:
                    dest[key] = []
                if store_data:
                    dest[key].extend(src[key][-1:])  # Only take the last entry from src[key]

    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        """Load episode data from the file."""
        # Not applicable for this handler since we store aggregated data
        raise NotImplementedError("Load episode not supported for preprocessed format")

    def flush(self, *, force: bool = False):
        """Atomically save pending data when due, or immediately if forced."""
        if not self._file_path or not self._dirty:
            return
        if not force and time.monotonic() - self._last_flush_time < self._flush_interval_s:
            return
        # Preserve the previous checkpoint if serialization fails or is interrupted.
        fd, temporary_path = tempfile.mkstemp(dir=os.path.dirname(self._file_path) or ".", suffix=".pt")
        try:
            with os.fdopen(fd, "wb") as output:
                torch.save({**self._episode_data, "quat_convention": "xyzw"}, output)
            os.replace(temporary_path, self._file_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        self._last_flush_time = time.monotonic()
        self._dirty = False

    def close(self):
        """Close the dataset file handler."""
        self.flush(force=True)
        self._episode_data = {}
        self._file_path = None

    def add_env_args(self, env_args: dict):
        pass
