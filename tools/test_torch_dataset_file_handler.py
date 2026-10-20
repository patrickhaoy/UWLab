# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only tests for periodic dataset checkpoints and final-save integrity."""

import tempfile
import torch
import unittest
from pathlib import Path
from unittest.mock import patch

from isaaclab.utils.datasets import EpisodeData

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler


class TestTorchDatasetFileHandler(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "resets.pt"

    def episode(self, value):
        episode = EpisodeData()
        episode.data = {"initial_state": {"position": torch.tensor([[value]], dtype=torch.float32)}}
        episode.success = True
        return episode

    def values(self):
        data = torch.load(self.path, weights_only=True)
        self.assertEqual(data["quat_convention"], "xyzw")
        return torch.stack(data["initial_state"]["position"]).flatten().tolist()

    def test_periodic_flush_and_final_save(self):
        handler = TorchDatasetFileHandler(flush_interval_s=60)
        handler.create(str(self.path))
        with patch("uwlab.utils.datasets.torch_dataset_file_handler.time.monotonic", return_value=100) as clock:
            handler.write_episode(self.episode(1))
            handler.flush()
            self.assertEqual(self.values(), [1])
            handler.write_episode(self.episode(2))
            clock.return_value = 159
            handler.flush()
            self.assertEqual(self.values(), [1])
            clock.return_value = 160
            handler.flush()
            self.assertEqual(self.values(), [1, 2])
            handler.write_episode(self.episode(3))
            handler.close()
            self.assertEqual(self.values(), [1, 2, 3])

    def test_default_saves_every_export_and_skips_unchanged_data(self):
        handler = TorchDatasetFileHandler()
        handler.create(str(self.path))
        for value in (1, 2):
            handler.write_episode(self.episode(value))
            handler.flush()
            self.assertEqual(len(self.values()), value)
        with patch("uwlab.utils.datasets.torch_dataset_file_handler.torch.save") as save:
            handler.flush()
            handler.close()
            save.assert_not_called()

    def test_failed_save_preserves_checkpoint_and_can_retry(self):
        handler = TorchDatasetFileHandler()
        handler.create(str(self.path))
        handler.write_episode(self.episode(1))
        handler.flush()
        original = self.path.read_bytes()
        handler.write_episode(self.episode(2))
        with patch("uwlab.utils.datasets.torch_dataset_file_handler.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                handler.flush()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        handler.close()
        self.assertEqual(self.values(), [1, 2])

    def test_reusing_handler_resets_checkpoint_timer(self):
        handler = TorchDatasetFileHandler(flush_interval_s=60)
        handler.create(str(self.path))
        handler.write_episode(self.episode(1))
        handler.close()
        handler.create(str(self.path))
        handler.write_episode(self.episode(2))
        handler.flush()
        self.assertEqual(self.values(), [2])

    def test_invalid_interval(self):
        for interval in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                TorchDatasetFileHandler(flush_interval_s=interval)


if __name__ == "__main__":
    unittest.main()
