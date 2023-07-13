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

from os import path

from mlrun import code_to_function, new_model_server
from tests.conftest import examples_path, results


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

    fn.run(workdir=str(examples_path))


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

    fn.run(handler="training", params={"p1": 5})
