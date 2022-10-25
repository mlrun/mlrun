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

from os import path

from mlrun import code_to_function, get_run_db, new_model_server
from mlrun.runtimes.function import compile_function_config
from mlrun.utils import get_in, parse_versioned_object_uri
from tests.conftest import examples_path, results, tests_root_directory


def test_job_nb():
    filename = f"{examples_path}/mlrun_jobs.ipynb"
    fn = code_to_function(filename=filename, kind="job")
    assert fn.kind == "job", "kind not set, test failed"
    assert fn.spec.build.functionSourceCode, "code not embedded"
    assert fn.spec.build.origin_filename == filename, "did not record filename"


def test_nuclio_nb():
    filename = f"{examples_path}/xgb_serving.ipynb"
    fn = new_model_server(
        "iris-srv",
        filename=filename,
        models={"iris_v1": "xyz"},
        model_class="XGBoostModel",
    )
    assert fn.kind == "remote", "kind not set, test failed"
    assert fn.spec.base_spec, "base_spec not found"


def test_nuclio_nb_serving():
    filename = "https://raw.githubusercontent.com/mlrun/mlrun/master/examples/xgb_serving.ipynb"
    fn = code_to_function(filename=filename)
    assert fn.kind == "remote", "kind not set, test failed"
    assert fn.spec.function_kind == "serving", "code not embedded"
    assert fn.spec.build.origin_filename == filename, "did not record filename"


def test_job_file():
    filename = f"{examples_path}/training.py"
    fn = code_to_function(filename=filename, kind="job")
    assert fn.kind == "job", "kind not set, test failed"
    assert fn.spec.build.functionSourceCode, "code not embedded"
    assert fn.spec.build.origin_filename == filename, "did not record filename"
    assert type(fn.metadata.labels) == dict, "metadata labels were not set"
    run = fn.run(workdir=str(examples_path), local=True, artifact_path="/tmp")

    project, uri, tag, hash_key = parse_versioned_object_uri(run.spec.function)
    local_fn = get_run_db().get_function(uri, project, tag=tag, hash_key=hash_key)
    assert local_fn["spec"]["command"] == filename, "wrong command path"
    assert (
        local_fn["spec"]["build"]["functionSourceCode"]
        == fn.spec.build.functionSourceCode
    ), "code was not copied to local function"


def test_job_file_noembed():
    name = f"{examples_path}/training.py"
    fn = code_to_function(filename=name, kind="job", embed_code=False)
    assert fn.kind == "job", "kind not set, test failed"
    assert not fn.spec.build.functionSourceCode, fn.spec.build.functionSourceCode
    assert fn.spec.command == name, "filename not set in command"


def test_local_file_noembed():
    name = f"{examples_path}/training.py"
    fn = code_to_function(filename=name, kind="local", embed_code=False)
    assert fn.kind == "local", "kind not set, test failed"
    assert not fn.spec.build.functionSourceCode, fn.spec.build.functionSourceCode
    assert fn.spec.command == name, "filename not set in command"

    fn.run(workdir=str(examples_path), artifact_path="/tmp")


def test_job_file_codeout():
    name = f"{examples_path}/mlrun_jobs.ipynb"
    out = f"{results}/ctf_tst.py"
    fn = code_to_function(filename=name, kind="job", code_output=out, embed_code=False)
    assert fn.kind == "job", "kind not set, test failed"
    assert not fn.spec.build.functionSourceCode, fn.spec.build.functionSourceCode
    assert fn.spec.command == out, "filename not set to out in command"
    assert path.isfile(out), "output not generated"


def test_local_file_codeout():
    name = f"{examples_path}/mlrun_jobs.ipynb"
    out = f"{results}/ctf_tst.py"
    fn = code_to_function(
        filename=name, kind="local", code_output=out, embed_code=False
    )
    assert fn.kind == "local", "kind not set, test failed"
    assert not fn.spec.build.functionSourceCode, fn.spec.build.functionSourceCode
    assert fn.spec.command == out, "filename not set to out in command"
    assert path.isfile(out), "output not generated"

    fn.run(handler="training", params={"p1": 5}, artifact_path="/tmp")


def test_nuclio_py():
    name = f"{examples_path}/training.py"
    fn = code_to_function("nuclio", filename=name, kind="nuclio", handler="my_hand")
    name, project, config = compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert get_in(config, "spec.build.functionSourceCode"), "no source code"
    assert get_in(config, "spec.runtime").startswith("py"), "runtime not set"
    assert get_in(config, "spec.handler") == "training:my_hand", "wrong handler"


def test_nuclio_golang():
    name = f"{tests_root_directory}/assets/hello.go"
    fn = code_to_function(
        "nuclio", filename=name, kind="nuclio", handler="main:Handler"
    )
    name, project, config = compile_function_config(fn)
    assert fn.kind == "remote", "kind not set, test failed"
    assert get_in(config, "spec.runtime") == "golang", "golang was not detected and set"
    assert get_in(config, "spec.handler") == "main:Handler", "wrong handler"
