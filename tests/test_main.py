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

from subprocess import PIPE, run
from sys import executable, stderr

import mlrun
from tests.conftest import examples_path, out_path, tests_root_directory


def exec_main(op, args):
    cmd = [executable, "-m", "mlrun", op]
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE, cwd=examples_path)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=stderr)
        raise Exception(out.stderr.decode("utf-8"))
    return out.stdout.decode("utf-8")


def exec_run(cmd, args, test):
    args = args + ["--name", test, "--dump", cmd]
    return exec_main("run", args)


def compose_param_list(params: dict, flag="-p"):
    composed_params = []
    for k, v in params.items():
        composed_params += [flag, f"{k}={v}"]
    return composed_params


def test_main_run_basic():
    out = exec_run(
        f"{examples_path}/training.py",
        compose_param_list(dict(p1=5, p2='"aaa"')),
        "test_main_run_basic",
    )
    print(out)
    assert out.find("state: completed") != -1, out


def test_main_run_hyper():
    out = exec_run(
        f"{examples_path}/training.py",
        compose_param_list(dict(p2=[4, 5, 6]), "-x"),
        "test_main_run_hyper",
    )
    print(out)
    assert out.find("state: completed") != -1, out
    assert out.find("iterations:") != -1, out


def test_main_run_noctx():
    out = exec_run(
        f"{tests_root_directory}/no_ctx.py",
        ["--mode", "noctx"] + compose_param_list(dict(p1=5, p2='"aaa"')),
        "test_main_run_noctx",
    )
    print(out)
    assert out.find("state: completed") != -1, out


def test_main_run_archive():
    args = f"--source {examples_path} --handler handler"
    out = exec_run("./myfunc.py", args.split(), "test_main_run_archive")
    assert out.find("state: completed") != -1, out


def test_main_local_source():
    args = f"--source {examples_path} --handler my_func"
    out = exec_run("./handler.py", args.split(), "test_main_local_source")
    print(out)
    assert out.find("state: completed") != -1, out


def test_main_run_archive_subdir():
    runtime = str({"spec": {"pythonpath": "./subdir"}})
    args = f'--source {examples_path}/archive.zip -r "{runtime}"'
    out = exec_run("./subdir/func2.py", args.split(), "test_main_run_archive_subdir")
    print(out)
    assert out.find("state: completed") != -1, out


def test_main_local_flag():
    fn = mlrun.code_to_function(
        filename=f"{examples_path}/handler.py", kind="job", handler="my_func"
    )
    yaml_path = f"{out_path}/myfunc.yaml"
    fn.export(yaml_path)
    args = f"-f {yaml_path} --local"
    out = exec_run("", args.split(), "test_main_local_flag")
    print(out)
    assert out.find("state: completed") != -1, out
