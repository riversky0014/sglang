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
"""Import smoke test for CPU serving without the optional AOT package."""

import os
import subprocess
import sys
import textwrap

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=15, suite="stage-a-unit-test-mlx")


class TestCPUWithoutAOTImport(CustomTestCase):
    def test_http_server_import_keeps_process_torch_compile_unchanged(self):
        env = os.environ.copy()
        env["SGLANG_USE_CPU_ENGINE"] = "1"
        env["SGLANG_USE_MLX"] = "0"
        script = textwrap.dedent("""
            import sys

            sys.modules["sgl_kernel"] = None

            import torch

            original_compile = torch.compile

            from sglang.kernels.optional import get_optional_sgl_kernel
            from sglang.srt.entrypoints.http_server import launch_server
            from sglang.srt.managers.scheduler import Scheduler

            assert Scheduler is not None
            assert launch_server is not None
            assert torch.compile is original_compile
            assert get_optional_sgl_kernel("assign_req_to_token_pool_cpu") is None
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    import unittest

    unittest.main()
