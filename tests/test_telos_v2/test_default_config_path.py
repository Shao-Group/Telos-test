"""Bundled default Stage I config is present."""

from __future__ import annotations

import unittest
from pathlib import Path

from telos_v2.config_loader import default_stage1_config_path


class TestDefaultStage1Config(unittest.TestCase):
    def test_default_yaml_exists(self) -> None:
        p = default_stage1_config_path()
        self.assertTrue(p.is_file(), msg=f"missing {p}")


if __name__ == "__main__":
    unittest.main()
