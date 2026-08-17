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
"""End-to-end smoke test for experimental macOS PyTorch CPU serving."""

import os
import platform
import sys
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

# This test deliberately runs the PyTorch CPU backend on the Apple Silicon
# runner. The MLX e2e suite is currently the repository's macOS ARM64 lane.
register_mlx_ci(est_time=180, suite="stage-b-e2e-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
MODEL_PATH = os.environ.get("SGLANG_MACOS_CPU_TEST_MODEL", "facebook/opt-125m")


@unittest.skipUnless(_IS_APPLE_SILICON, "requires an Apple Silicon macOS runner")
class TestMacOSPyTorchCPUServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env["SGLANG_USE_CPU_ENGINE"] = "1"
        env["SGLANG_USE_MLX"] = "0"
        env["PATH"] = os.pathsep.join(
            [os.path.dirname(sys.executable), env.get("PATH", "")]
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "cpu",
                "--attention-backend",
                "torch_native",
                "--tp-size",
                "1",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-overlap-schedule",
                "--mem-fraction-static",
                "0.1",
                "--max-running-requests",
                "1",
                "--context-length",
                "512",
            ],
            env=env,
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_deterministic_generation(self):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }

        first = requests.post(f"{self.base_url}/generate", json=payload, timeout=120)
        first.raise_for_status()
        second = requests.post(f"{self.base_url}/generate", json=payload, timeout=120)
        second.raise_for_status()

        first_text = first.json()["text"]
        second_text = second.json()["text"]
        self.assertIsInstance(first_text, str)
        self.assertTrue(first_text)
        self.assertEqual(first_text, second_text)


if __name__ == "__main__":
    unittest.main()
