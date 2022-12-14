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
import datetime
import os
import pathlib
import sys
import traceback
from base64 import b64encode
from subprocess import PIPE, run
from sys import executable, stderr

import pytest

import mlrun
from tests.conftest import examples_path, out_path, tests_root_directory


def exec_main(op, args, cwd=examples_path, raise_on_error=True):
    cmd = [executable, "-m", "mlrun", op]
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=stderr)
        print(out.stdout.decode("utf-8"), file=stderr)
        print(traceback.format_exc())
        if raise_on_error:
            raise Exception(out.stderr.decode("utf-8"))
        else:
            return out.stderr.decode("utf-8")

    return out.stdout.decode("utf-8")


def exec_run(cmd, args, test, raise_on_error=True):
    args = args + ["--name", test, "--dump", cmd]
    return exec_main("run", args, raise_on_error=raise_on_error)


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


def test_main_run_wait_for_completion():
    """
    Test that the run command waits for the run to complete before returning
    (mainly sanity as this is expected when running local function)
    """
    path = str(pathlib.Path(__file__).absolute().parent / "assets" / "sleep.py")
    time_to_sleep = 10
    start_time = datetime.datetime.now()
    out = exec_run(
        path,
        compose_param_list(dict(time_to_sleep=time_to_sleep))
        + ["--handler", "handler"],
        "test_main_run_wait_for_completion",
    )
    end_time = datetime.datetime.now()
    print(out)
    assert out.find("state: completed") != -1, out
    assert (
        end_time - start_time
    ).seconds >= time_to_sleep, "run did not wait for completion"


def test_main_run_hyper():
    out = exec_run(
        f"{examples_path}/training.py",
        compose_param_list(dict(p2=[4, 5, 6]), "-x"),
        "test_main_run_hyper",
    )
    print(out)
    assert out.find("state: completed") != -1, out
    assert out.find("iterations:") != -1, out


def test_main_run_args():
    out = exec_run(
        f"{tests_root_directory}/no_ctx.py -x " + "{p2}",
        ["--uid", "123457"] + compose_param_list(dict(p1=5, p2="aaa")),
        "test_main_run_args",
    )
    print(out)
    assert out.find("state: completed") != -1, out
    db = mlrun.get_run_db()
    state, log = db.get_log("123457")
    print(log)
    assert str(log).find(", -x, aaa") != -1, "params not detected in argv"


@pytest.mark.parametrize(
    "op,args,raise_on_error,expected_output",
    [
        # bad flag before command
        [
            "run",
            [
                "--bad-flag",
                "--name",
                "test_main_run_basic",
                "--dump",
                f"{examples_path}/training.py",
            ],
            False,
            "Error: Invalid value for '[URL]': URL (--bad-flag) cannot start with '-', "
            "ensure the command options are typed correctly. Preferably use '--' to separate options and "
            "arguments e.g. 'mlrun run --option1 --option2 -- [URL] [--arg1|arg1] [--arg2|arg2]'",
        ],
        # bad flag with no command
        [
            "run",
            ["--name", "test_main_run_basic", "--bad-flag"],
            False,
            "Error: Invalid value for '[URL]': URL (--bad-flag) cannot start with '-', "
            "ensure the command options are typed correctly. Preferably use '--' to separate options and "
            "arguments e.g. 'mlrun run --option1 --option2 -- [URL] [--arg1|arg1] [--arg2|arg2]'",
        ],
        # bad flag after -- separator
        [
            "run",
            ["--name", "test_main_run_basic", "--", "-notaflag"],
            False,
            "Error: Invalid value for '[URL]': URL (-notaflag) cannot start with '-', "
            "ensure the command options are typed correctly. Preferably use '--' to separate options and "
            "arguments e.g. 'mlrun run --option1 --option2 -- [URL] [--arg1|arg1] [--arg2|arg2]'",
        ],
        # correct command with -- separator
        [
            "run",
            [
                "--name",
                "test_main_run_basic",
                "--",
                f"{examples_path}/training.py",
                "--some-arg",
            ],
            True,
            "status=completed",
        ],
    ],
)
def test_main_run_args_validation(op, args, raise_on_error, expected_output):
    out = exec_main(
        op,
        args,
        raise_on_error=raise_on_error,
    )
    print(out)
    assert out.find(expected_output) != -1, out


code = """
import mlrun, sys
if __name__ == "__main__":
    context = mlrun.get_or_create_ctx("test1")
    context.log_result("my_args", sys.argv)
    context.commit(completed=True)
"""


def test_main_run_args_from_env():
    os.environ["MLRUN_EXEC_CODE"] = b64encode(code.encode("utf-8")).decode("utf-8")
    os.environ["MLRUN_EXEC_CONFIG"] = (
        '{"spec":{"parameters":{"x": "bbb"}},'
        '"metadata":{"uid":"123459", "name":"tst", "labels": {"kind": "job"}}}'
    )

    out = exec_run(
        "'main.py -x {x}'",
        ["--from-env"],
        "test_main_run_args_from_env",
    )
    db = mlrun.get_run_db()
    run = db.read_run("123459")
    print(out)
    assert run["status"]["state"] == "completed", out
    assert run["status"]["results"]["my_args"] == [
        "main.py",
        "-x",
        "bbb",
    ], "params not detected in argv"


nonpy_code = """
echo "abc123" $1
"""


@pytest.mark.skipif(sys.platform == "win32", reason="skip on windows")
def test_main_run_nonpy_from_env():
    os.environ["MLRUN_EXEC_CODE"] = b64encode(nonpy_code.encode("utf-8")).decode(
        "utf-8"
    )
    os.environ[
        "MLRUN_EXEC_CONFIG"
    ] = '{"spec":{},"metadata":{"uid":"123411", "name":"tst", "labels": {"kind": "job"}}}'

    # --kfp flag will force the logs to print (for the assert)
    out = exec_run(
        "bash {codefile} xx",
        ["--from-env", "--mode", "pass", "--kfp"],
        "test_main_run_nonpy_from_env",
    )
    db = mlrun.get_run_db()
    run = db.read_run("123411")
    assert run["status"]["state"] == "completed", out
    state, log = db.get_log("123411")
    print(state, log)
    assert str(log).find("abc123 xx") != -1, "incorrect output"


def test_main_run_pass():
    out = exec_run(
        "python -c print(56)",
        ["--mode", "pass", "--uid", "123458"],
        "test_main_run_pass",
    )
    print(out)
    assert out.find("state: completed") != -1, out
    db = mlrun.get_run_db()
    state, log = db.get_log("123458")
    assert str(log).find("56") != -1, "incorrect output"


def test_main_run_pass_args():
    out = exec_run(
        "'python -c print({x})'",
        ["--mode", "pass", "--uid", "123451", "-p", "x=33"],
        "test_main_run_pass",
    )
    print(out)
    assert out.find("state: completed") != -1, out
    db = mlrun.get_run_db()
    state, log = db.get_log("123451")
    print(log)
    assert str(log).find("33") != -1, "incorrect output"


def test_main_run_archive():
    args = f"--source {examples_path}/archive.zip --handler handler -p p1=1"
    out = exec_run("./myfunc.py", args.split(), "test_main_run_archive")
    assert out.find("state: completed") != -1, out


def test_main_local_source():
    args = f"--source {examples_path} --handler my_func"
    with pytest.raises(Exception) as e:
        exec_run("./handler.py", args.split(), "test_main_local_source")
    assert (
        "source must be a compressed (tar.gz / zip) file, a git repo, a file path or in the project's context (.)"
        in str(e.value)
    )


def test_main_run_archive_subdir():
    runtime = '{"spec":{"pythonpath":"./subdir"}}'
    args = f"--source {examples_path}/archive.zip -r {runtime}"
    out = exec_run("./subdir/func2.py", args.split(), "test_main_run_archive_subdir")
    print(out)
    assert out.find("state: completed") != -1, out


def test_main_local_project():
    project_path = str(pathlib.Path(__file__).parent / "assets")
    args = "-f simple -p x=2 --dump"
    out = exec_main("run", args.split(), cwd=project_path)
    assert out.find("state: completed") != -1, out
    assert out.find("y: 4") != -1, out  # y = x * 2


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


def test_main_run_class():
    function_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")

    out = exec_run(
        function_path,
        compose_param_list(dict(x=8)) + ["--handler", "mycls::mtd"],
        "test_main_run_class",
    )
    assert out.find("state: completed") != -1, out
    assert out.find("rx: 8") != -1, out


def test_run_from_module():
    args = ["--name", "test1", "--dump", "--handler", "json.dumps", "-p", "obj=[6,7]"]
    out = exec_main("run", args)
    assert out.find("state: completed") != -1, out
    assert out.find("return: '[6, 7]'") != -1, out


def test_main_env_file():
    # test run with env vars loaded from a .env file
    function_path = str(pathlib.Path(__file__).parent / "assets" / "handler.py")
    envfile = str(pathlib.Path(__file__).parent / "assets" / "envfile")

    out = exec_run(
        function_path,
        ["--handler", "env_file_test", "--env-file", envfile],
        "test_main_env_file",
    )
    assert out.find("state: completed") != -1, out
    assert out.find("ENV_ARG1: '123'") != -1, out
    assert out.find("kfp_ttl: 12345") != -1, out
