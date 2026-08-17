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
"""Tests for CPU distributed initialization without AOT collectives."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import bootstrap
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestCPUWithoutAOTBootstrap(CustomTestCase):
    def test_tp_one_continues_but_tp_two_is_rejected(self):
        missing_ops = SimpleNamespace(sgl_kernel=SimpleNamespace())
        with (
            patch.object(bootstrap, "_is_cpu_arm64", True),
            patch.object(bootstrap, "_is_cpu_amx_available", False),
            patch.object(bootstrap.sys, "platform", "darwin"),
            patch.object(bootstrap.torch, "ops", missing_ops),
        ):
            bootstrap._init_cpu_threads_env(tp_size=1, tp_rank=0, local_omp_cpuid=None)
            with self.assertRaisesRegex(RuntimeError, "use --tp-size 1"):
                bootstrap._init_cpu_threads_env(
                    tp_size=2, tp_rank=0, local_omp_cpuid=None
                )

    def test_linux_keeps_aot_dependency(self):
        missing_ops = SimpleNamespace(sgl_kernel=SimpleNamespace())
        with (
            patch.object(bootstrap, "_is_cpu_arm64", True),
            patch.object(bootstrap, "_is_cpu_amx_available", False),
            patch.object(bootstrap.sys, "platform", "linux"),
            patch.object(bootstrap.torch, "ops", missing_ops),
            self.assertRaises(AttributeError),
        ):
            bootstrap._init_cpu_threads_env(tp_size=1, tp_rank=0, local_omp_cpuid=None)


if __name__ == "__main__":
    import unittest

    unittest.main()
