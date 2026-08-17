# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for portable CPU topology discovery."""

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

import unittest
from unittest.mock import patch

from sglang.srt.utils.common import (
    get_physical_cpus_by_numa,
    parse_lscpu_topology,
)
from sglang.test.test_utils import CustomTestCase


class _ProcessWithoutAffinity:
    pass


class TestMacOSTopology(CustomTestCase):
    @patch("sglang.srt.utils.common.psutil.cpu_count", return_value=4)
    @patch("sglang.srt.utils.common.platform.system", return_value="Darwin")
    def test_darwin_is_represented_as_one_uma_node(self, _mock_system, _mock_cpu_count):
        self.assertEqual(
            parse_lscpu_topology(),
            [(0, 0, 0, 0), (1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0)],
        )

    @patch(
        "sglang.srt.utils.common.psutil.cpu_count",
        side_effect=[None, 2],
    )
    @patch("sglang.srt.utils.common.platform.system", return_value="Darwin")
    def test_darwin_falls_back_to_logical_cpu_count(
        self, _mock_system, _mock_cpu_count
    ):
        self.assertEqual(
            parse_lscpu_topology(),
            [(0, 0, 0, 0), (1, 1, 0, 0)],
        )

    @patch(
        "sglang.srt.utils.common.parse_lscpu_topology",
        return_value=[(0, 0, 0, 0), (1, 1, 0, 0), (2, 2, 0, 0)],
    )
    @patch(
        "sglang.srt.utils.common.psutil.Process",
        return_value=_ProcessWithoutAffinity(),
    )
    def test_missing_affinity_uses_all_discovered_cpus(
        self, _mock_process, _mock_topology
    ):
        self.assertEqual(get_physical_cpus_by_numa(), {0: {0, 1, 2}})


if __name__ == "__main__":
    unittest.main()
