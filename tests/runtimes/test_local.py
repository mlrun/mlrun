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
from unittest.mock import MagicMock, patch
import mlrun.model
import mlrun.runtimes.local


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


@pytest.mark.parametrize(
    "return_value",
    [0, 0.0, False, "", [], {}],
    ids=["zero_int", "zero_float", "false", "empty_string", "empty_list", "empty_dict"],
)
def test_exec_from_params_logs_falsy_return_values(return_value):
    """
    Verify that exec_from_params logs handler return values even when
    they are falsy (0, False, empty string, etc.). Previously, the check
    used ``if val:`` which silently dropped these legitimate results.
    """

    def handler():
        return return_value

    runobj = MagicMock(spec=mlrun.model.RunObject)
    runobj.spec.verbose = False
    runobj.spec.inputs_type_hints = {}
    runobj.spec.returns = []
    runobj.spec.parameters = {}
    runobj.spec.inputs = {}

    context = MagicMock()
    context._parameters = {}
    context._reset_on_run = False

    with patch("mlrun.mlconf") as mock_conf:
        mock_conf.packagers.enabled = False
        mlrun.runtimes.local.exec_from_params(handler, runobj, context)

    context.log_result.assert_called_once_with("return", return_value)


def test_exec_from_params_does_not_log_none_return():
    """
    Verify that exec_from_params does NOT log a return result when the
    handler returns None (the default when no return statement is used).
    """

    def handler():
        return None

    runobj = MagicMock(spec=mlrun.model.RunObject)
    runobj.spec.verbose = False
    runobj.spec.inputs_type_hints = {}
    runobj.spec.returns = []
    runobj.spec.parameters = {}
    runobj.spec.inputs = {}

    context = MagicMock()
    context._parameters = {}
    context._reset_on_run = False

    with patch("mlrun.mlconf") as mock_conf:
        mock_conf.packagers.enabled = False
        mlrun.runtimes.local.exec_from_params(handler, runobj, context)

    context.log_result.assert_not_called()
