import pytest

import mlrun
import mlrun.api.crud.runtimes.nuclio.function
from tests.conftest import examples_path


def test_compiled_function_config_nuclio_golang():
    name = f"{examples_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    (
        name,
        project,
        config,
    ) = mlrun.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert mlrun.utils.get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert mlrun.utils.get_in(config, "spec.runtime").startswith(
        "py"
    ), "runtime not set"
    assert (
        mlrun.utils.get_in(config, "spec.handler") == "training:my_hand"
    ), "wrong handler"


def test_compiled_function_config_nuclio_python():
    name = f"{examples_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    (
        name,
        project,
        config,
    ) = mlrun.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert mlrun.utils.get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert mlrun.utils.get_in(config, "spec.runtime").startswith(
        "py"
    ), "runtime not set"
    assert (
        mlrun.utils.get_in(config, "spec.handler") == "training:my_hand"
    ), "wrong handler"


@pytest.mark.parametrize(
    "handler, expected",
    [
        (None, ("", "main:handler")),
        ("x", ("", "x:handler")),
        ("x:y", ("", "x:y")),
        ("dir#", ("dir", "main:handler")),
        ("dir#x", ("dir", "x:handler")),
        ("dir#x:y", ("dir", "x:y")),
    ],
)
def test_resolve_work_dir_and_handler(handler, expected):
    assert (
        expected
        == mlrun.api.crud.runtimes.nuclio.function._resolve_work_dir_and_handler(
            handler
        )
    )


@pytest.mark.parametrize(
    "mlrun_client_version,python_version,expected_runtime",
    [
        ("1.3.0", "3.9.16", "python:3.9"),
        ("1.3.0", "3.7.16", "python:3.7"),
        (None, None, "python:3.7"),
        (None, "3.9.16", "python:3.7"),
        ("1.3.0", None, "python:3.7"),
        ("0.0.0-unstable", "3.9.16", "python:3.9"),
        ("0.0.0-unstable", "3.7.16", "python:3.7"),
        ("1.2.0", "3.9.16", "python:3.7"),
        ("1.2.0", "3.7.16", "python:3.7"),
    ],
)
def test_resolve_nuclio_runtime_python_image(
    mlrun_client_version, python_version, expected_runtime
):
    assert (
        expected_runtime
        == mlrun.api.crud.runtimes.nuclio.function._resolve_nuclio_runtime_python_image(
            mlrun_client_version, python_version
        )
    )
