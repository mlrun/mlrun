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
import unittest.mock
from unittest.mock import MagicMock, patch

import pytest

import mlrun.model
import mlrun.runtimes.local
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


def test_custom_env_vars_merged_into_subprocess_env():
    """Verify that custom env dict entries (e.g. PYTHONPATH, MLRUN_LOG_LEVEL)
    are present in the environment passed to Popen, merged with os.environ."""
    custom_env = {
        "PYTHONPATH": "/custom/path",
        "MLRUN_LOG_LEVEL": "DEBUG",
    }

    with unittest.mock.patch("mlrun.runtimes.local.Popen") as mock_popen:
        mock_process = unittest.mock.MagicMock()
        mock_process.stdout.readline.return_value = ""
        mock_process.stderr.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        run_exec(["echo", "hello"], args=None, env=custom_env)

        # Popen should have been called once
        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args
        passed_env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")

        # The custom env vars must be present in the passed environment
        assert passed_env is not None
        assert passed_env["PYTHONPATH"] == "/custom/path"
        assert passed_env["MLRUN_LOG_LEVEL"] == "DEBUG"

        # os.environ entries should also be present (merged)
        assert "PATH" in passed_env

def test_no_custom_env_uses_os_environ():
    """When env=None, run_exec should still pass a copy of os.environ."""
    with unittest.mock.patch("mlrun.runtimes.local.Popen") as mock_popen:
        mock_process = unittest.mock.MagicMock()
        mock_process.stdout.readline.return_value = ""
        mock_process.stderr.readline.return_value = ""
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        run_exec(["echo", "hello"], args=None, env=None)

        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args
        passed_env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")

        # Should still have PATH from os.environ
        assert passed_env is not None
        assert "PATH" in passed_env
