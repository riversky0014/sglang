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
"""Tests for resolving optional symbols from the ``sgl_kernel`` package."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels import optional
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestOptionalSglKernel(CustomTestCase):
    def test_absent_package_is_optional_only_on_macos_arm64(self):
        error = ModuleNotFoundError("No module named 'sgl_kernel'", name="sgl_kernel")
        with (
            patch.object(optional.importlib, "import_module", side_effect=error),
            patch.object(optional.sys, "platform", "darwin"),
            patch.object(optional.platform, "machine", return_value="arm64"),
        ):
            self.assertIsNone(optional.get_optional_sgl_kernel("missing"))

        with (
            patch.object(optional.importlib, "import_module", side_effect=error),
            patch.object(optional.sys, "platform", "linux"),
            self.assertRaises(ModuleNotFoundError),
        ):
            optional.get_optional_sgl_kernel("missing")

    def test_internal_import_failure_is_not_hidden(self):
        error = ModuleNotFoundError("No module named 'dependency'", name="dependency")
        with patch.object(optional.importlib, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as context:
                optional.get_optional_sgl_kernel("missing")

        self.assertEqual(context.exception.name, "dependency")

    def test_missing_symbol_is_optional_only_on_macos_arm64(self):
        kernel = lambda: None
        module = SimpleNamespace(kernel=kernel)
        with patch.object(optional.importlib, "import_module", return_value=module):
            self.assertIs(optional.get_optional_sgl_kernel("kernel"), kernel)

            with (
                patch.object(optional.sys, "platform", "darwin"),
                patch.object(optional.platform, "machine", return_value="arm64"),
            ):
                self.assertIsNone(optional.get_optional_sgl_kernel("missing"))

            with (
                patch.object(optional.sys, "platform", "linux"),
                self.assertRaises(AttributeError),
            ):
                optional.get_optional_sgl_kernel("missing")


if __name__ == "__main__":
    import unittest

    unittest.main()
