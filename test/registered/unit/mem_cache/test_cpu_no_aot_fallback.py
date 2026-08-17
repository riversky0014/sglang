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
"""Correctness tests for CPU cache fallbacks without optional AOT kernels."""

from unittest.mock import patch

import torch

from sglang.srt.mem_cache import allocation, memory_pool
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")


class TestCPUWithoutAOTCacheFallback(CustomTestCase):
    def test_request_mapping_uses_torch_fallback(self):
        req_to_token = torch.full((3, 8), -1, dtype=torch.int32)

        with (
            patch.object(allocation, "_is_cpu", True),
            patch.object(allocation, "_assign_req_to_token_pool_cpu", None),
        ):
            allocation.assign_req_to_token_pool_func(
                torch.tensor([2, 0]),
                req_to_token,
                torch.tensor([1, 3]),
                torch.tensor([3, 6]),
                torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64),
                2,
            )

        self.assertEqual(req_to_token[2, 1:3].tolist(), [10, 11])
        self.assertEqual(req_to_token[0, 3:6].tolist(), [20, 21, 22])

    def test_request_mapping_keeps_available_aot_path(self):
        calls = []

        def fake_kernel(*args):
            calls.append(args[-1])

        with (
            patch.object(allocation, "_is_cpu", True),
            patch.object(allocation, "_assign_req_to_token_pool_cpu", fake_kernel),
        ):
            allocation.assign_req_to_token_pool_func(
                torch.tensor([0]),
                torch.full((1, 4), -1, dtype=torch.int32),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([7], dtype=torch.int32),
                1,
            )

        self.assertEqual(calls, [4])

    def test_kv_move_uses_native_torch_fallback(self):
        with (patch.object(memory_pool, "_is_macos_arm64_cpu", True),):
            pool = memory_pool.MHATokenToKVPool(
                size=8,
                page_size=1,
                dtype=torch.float32,
                head_num=1,
                head_dim=2,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
                enable_kv_cache_copy=True,
            )

        self.assertTrue(pool.use_native_move_kv_cache)
        key = pool.get_key_buffer(0)
        value = pool.get_value_buffer(0)
        key[2].fill_(3)
        value[2].fill_(5)
        pool.move_kv_cache(torch.tensor([4]), torch.tensor([2]))
        torch.testing.assert_close(key[4], key[2])
        torch.testing.assert_close(value[4], value[2])


if __name__ == "__main__":
    import unittest

    unittest.main()
