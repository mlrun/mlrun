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

from tests.conftest import (
    examples_path,
    out_path,
    tag_test,
    verify_state,
)
from mlrun import new_task, run_local, code_to_function

base_spec = new_task(params={"p1": 8}, out_path=out_path)
base_spec.spec.inputs = {"infile.txt": "infile.txt"}


def test_run_local():
    spec = tag_test(base_spec, "test_run_local")
    result = run_local(
        spec, command="{}/training.py".format(examples_path), workdir=examples_path
    )
    verify_state(result)


def test_run_local_handler():
    spec = tag_test(base_spec, "test_run_local_handler")
    spec.spec.handler = "my_func"
    result = run_local(
        spec, command="{}/handler.py".format(examples_path), workdir=examples_path
    )
    verify_state(result)


def test_run_local_nb():
    spec = tag_test(base_spec, "test_run_local_handler")
    spec.spec.handler = "training"
    result = run_local(
        spec, command="{}/mlrun_jobs.ipynb".format(examples_path), workdir=examples_path
    )
    verify_state(result)


def test_run_local_yaml():
    spec = tag_test(base_spec, "test_run_local_handler")
    spec.spec.handler = "training"
    nbpath = "{}/mlrun_jobs.ipynb".format(examples_path)
    ymlpath = path.join(out_path, "nbyaml.yaml")
    print("out path:", out_path, ymlpath)
    code_to_function(filename=nbpath, kind="job").export(ymlpath)
    result = run_local(spec, command=ymlpath, workdir=out_path)
    verify_state(result)


def test_run_local_obj():
    spec = tag_test(base_spec, "test_run_local_handler")
    spec.spec.handler = "training"
    nbpath = "{}/mlrun_jobs.ipynb".format(examples_path)
    ymlpath = path.join(out_path, "nbyaml.yaml")
    print("out path:", out_path, ymlpath)
    fn = code_to_function(filename=nbpath, kind="job").export(ymlpath)
    result = run_local(spec, command=fn, workdir=out_path)
    verify_state(result)
