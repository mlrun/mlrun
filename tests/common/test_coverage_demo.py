# Copyright 2026 Iguazio
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

import importlib.util
import pathlib

_path = (
    pathlib.Path(__file__).resolve().parents[2] / "mlrun" / "utils" / "coverage_demo.py"
)
spec = importlib.util.spec_from_file_location("mlrun.utils.coverage_demo", _path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)


def test_add():
    assert _mod.add(2, 3) == 5
