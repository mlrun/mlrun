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
#

import pytest

import mlrun
import server.api.crud.runtimes.nuclio.function
import server.api.crud.runtimes.nuclio.helpers
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
    ) = server.api.crud.runtimes.nuclio.function._compile_function_config(fn)
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
    ) = server.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert mlrun.utils.get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert mlrun.utils.get_in(config, "spec.runtime").startswith(
        "py"
    ), "runtime not set"
    assert (
        mlrun.utils.get_in(config, "spec.handler") == "training:my_hand"
    ), "wrong handler"


def test_compiled_function_config_sidecar_image_enrichment():
    mlrun.mlconf.httpdb.builder.docker_registry = "docker.io"
    name = f"{examples_path}/training.py"
    fn = mlrun.code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="my_hand"
    )
    fn.with_sidecar("my-sidecar", ".mlrun/mlrun")
    (
        name,
        project,
        config,
    ) = server.api.crud.runtimes.nuclio.function._compile_function_config(fn)
    assert mlrun.utils.get_in(config, "spec.sidecars"), "No sidecars"
    assert (
        mlrun.utils.get_in(config, "spec.sidecars")[0]["image"]
        == "docker.io/mlrun/mlrun:unstable"
    ), "Image not enriched"


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
        == server.api.crud.runtimes.nuclio.helpers.resolve_work_dir_and_handler(handler)
    )


@pytest.mark.parametrize(
    "mlrun_client_version,python_version,expected_runtime",
    [
        ("1.3.0", "3.9.16", "python:3.9"),
        ("1.3.0", "3.7.16", "python:3.7"),
        (None, None, mlrun.mlconf.default_nuclio_runtime),
        (None, "3.9.16", mlrun.mlconf.default_nuclio_runtime),
        ("1.3.0", None, mlrun.mlconf.default_nuclio_runtime),
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
        == server.api.crud.runtimes.nuclio.helpers.resolve_nuclio_runtime_python_image(
            mlrun_client_version, python_version
        )
    )
