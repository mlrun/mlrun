# Copyright 2023 Iguazio
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
import pathlib

import pytest

from mlrun.runtimes.local import run_exec


def test_run_exec_basic():
    out, err = run_exec(["echo"], ["hello"])
    assert out == "hello\n"
    assert err == ""


# ML-3710
@pytest.mark.parametrize("return_code", [0, 1])
def test_run_exec_verbose_stderr(return_code):
    script_path = str(
        pathlib.Path(__file__).parent
        / "assets"
        / f"verbose_stderr_return_code_{return_code}.py"
    )
    out, err = run_exec(["python"], [script_path])
    assert out == "some output\n"
    expected_err_length = 100000 if return_code else 0
    assert len(err) == expected_err_length
