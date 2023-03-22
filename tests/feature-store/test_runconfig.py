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
#
import pathlib

import mlrun
from mlrun.feature_store.common import RunConfig

function_path = str(pathlib.Path(__file__).parent / "assets" / "function.py")


def test_none_config():
    fn = RunConfig().to_function("serving", "x/y")
    assert fn.kind == "serving"
    assert fn.spec.image == "x/y"
    assert (
        fn.spec.build.functionSourceCode
    ), "serving source is empty (should have footer)"

    fn = RunConfig(image="a/b").to_function("job", "x/y")
    assert fn.kind == "job"
    assert fn.spec.image == "a/b"
    assert not fn.spec.build.functionSourceCode, "serving source is not empty"


def test_from_code():
    run_function = RunConfig(function_path, requirements=["x"]).to_function(
        "serving", "x/y"
    )
    code_function = mlrun.code_to_function(
        filename=function_path, kind="serving", requirements=["x"]
    )

    assert run_function.kind == "serving"
    assert (
        run_function.spec.build.functionSourceCode
        == code_function.spec.build.functionSourceCode
    )
    assert run_function.spec.build.commands == code_function.spec.build.commands


def test_from_func():
    code_function = mlrun.code_to_function(
        filename=function_path, kind="job", image="a/b"
    )
    run_function = RunConfig(code_function).to_function("serving", "x/y")

    assert run_function.kind == "job"
    assert run_function.spec.image == "a/b"
    assert (
        run_function.spec.build.functionSourceCode
        == code_function.spec.build.functionSourceCode
    )
