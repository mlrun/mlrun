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

import mlrun
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


def test_pre_run_points_command_at_extracted_module(tmp_path, monkeypatch):
    """`LocalRuntime._pre_run` points spec.command at the extracted module
    file and strips the module prefix from the run handler when the source
    is loaded at runtime (e.g. store:// CodeArtifact, git, archive).

    Without this, _get_handler would call load_module(file_name="",
    handler="module:func", ...), produce no module, and fail downstream
    with ModuleNotFoundError.
    """
    # Pre-create the module file the helper will look for.
    target_dir = tmp_path / "extracted"
    target_dir.mkdir()
    handler_path = target_dir / "handler.py"
    handler_path.write_text("def my_func(context):\n    return 42\n")

    monkeypatch.setattr(
        "mlrun.runtimes.local.extract_source",
        lambda *args, **kwargs: str(target_dir),
    )

    runtime = mlrun.runtimes.local.LocalRuntime()
    runtime.spec.build.source = "store://artifacts/proj/handler_code"
    run = mlrun.run.RunObject()
    run.spec.handler = "handler:my_func"
    execution = mlrun.run.MLClientCtx.from_dict(run.to_dict(), autocommit=False)

    runtime._pre_run(run, execution)

    assert runtime.spec.command == str(handler_path)
    assert run.spec.handler == "my_func"


def test_pre_run_warns_when_module_not_in_extracted_dir(tmp_path, monkeypatch):
    """When `module:func` is given but `<module>.py` doesn't exist in the
    extracted source dir, _pre_run logs a warning instead of silently
    falling through. Pinning this prevents the regression where a handler
    typo produces a cryptic ImportError far from the entry point.
    """
    target_dir = tmp_path / "extracted"
    target_dir.mkdir()
    # Note: NO handler.py — the helper should warn about the missing file.

    monkeypatch.setattr(
        "mlrun.runtimes.local.extract_source",
        lambda *args, **kwargs: str(target_dir),
    )

    # Capture logger.warning directly. caplog can miss records when the mlrun
    # logger's propagate flag is toggled by other test setup, so observe the
    # call site instead of relying on stdlib logging propagation.
    warn_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        mlrun.runtimes.local.logger,
        "warning",
        lambda msg, *args, **kwargs: warn_calls.append((msg, kwargs)),
    )

    runtime = mlrun.runtimes.local.LocalRuntime()
    runtime.spec.build.source = "store://artifacts/proj/missing_code"
    run = mlrun.run.RunObject()
    run.spec.handler = "missing_module:my_func"
    execution = mlrun.run.MLClientCtx.from_dict(run.to_dict(), autocommit=False)

    runtime._pre_run(run, execution)

    assert any(
        "module:func handler refers to a module that wasn't found" in msg
        for msg, _ in warn_calls
    )
    # spec.command stays unset; downstream handler logic raises with a less
    # cryptic error after the warning has been logged.
    assert not runtime.spec.command
    assert run.spec.handler == "missing_module:my_func"
