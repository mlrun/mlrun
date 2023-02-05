# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest.mock
import warnings

import mlrun


def test_deprecated_decorator_warning_is_shown():
    mlrun._mount_v3io_extended = unittest.mock.MagicMock()
    with warnings.catch_warnings(record=True) as w:
        mlrun.mount_v3io_legacy()

        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)
        assert (
            "Call to deprecated function (or staticmethod) mount_v3io_legacy. "
            "('mount_v3io_legacy' will be removed in 1.5.0, use 'mount_v3io' instead) -- "
            "Deprecated since version 1.3.0." in str(w[-1].message)
        )


def test_deprecation_warning_is_shown():
    mlrun.get_or_create_project = unittest.mock.MagicMock()
    mlrun.mlconf.artifact_path = "mock"
    with warnings.catch_warnings(record=True) as w:
        mlrun.set_environment(project="mock")

        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)
        assert (
            "'user_project' and 'project' are deprecated in 1.3.0, and will be removed in 1.5.0, use project "
            "APIs such as 'get_or_create_project', 'load_project' to configure the active project."
            in str(w[-1].message)
        )
