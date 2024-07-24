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
import datetime
import os
import pathlib
import sys
import tempfile
import traceback
from base64 import b64encode
from subprocess import run
from sys import executable, stderr

import pytest

import mlrun
import tests.integration.sdk_api.base
from tests.conftest import examples_path, out_path, tests_root_directory

code = """
import mlrun, sys
if __name__ == "__main__":
    context = mlrun.get_or_create_ctx("test1")
    context.log_result("my_args", sys.argv)
    context.commit(completed=True)
"""

nonpy_code = """
echo "abc123" $1
"""


class TestMain(tests.integration.sdk_api.base.TestMLRunIntegration):
    assets_path = (
        pathlib.Path(__file__).absolute().parent.parent.parent.parent / "run" / "assets"
    )

    def custom_setup(self):
        # ensure default project exists
        mlrun.get_or_create_project("default", allow_cross_project=True)

    def test_main_run_basic(self):
        out = self._exec_run(
            f"{examples_path}/training.py",
            self._compose_param_list(dict(p1=5, p2='"aaa"')),
            "test_main_run_basic",
        )
        print(out)
        assert out.find("state: completed") != -1, out

    def test_main_run_wait_for_completion(self):
        """
        Test that the run command waits for the run to complete before returning
        (mainly sanity as this is expected when running local function)
        """
        path = str(self.assets_path / "sleep.py")
        time_to_sleep = 10
        start_time = datetime.datetime.now()
        out = self._exec_run(
            path,
            self._compose_param_list(dict(time_to_sleep=time_to_sleep))
            + ["--handler", "handler"],
            "test_main_run_wait_for_completion",
        )
        end_time = datetime.datetime.now()
        print(out)
        assert out.find("state: completed") != -1, out
        assert (
            end_time - start_time
        ).seconds >= time_to_sleep, "run did not wait for completion"

    def test_main_run_hyper(self):
        out = self._exec_run(
            f"{examples_path}/training.py",
            self._compose_param_list(dict(p2=[4, 5, 6]), "-x"),
            "test_main_run_hyper",
        )
        print(out)
        assert out.find("state: completed") != -1, out
        assert out.find("iterations:") != -1, out

    def test_main_run_args(self):
        out = self._exec_run(
            f"{tests_root_directory}/no_ctx.py -x " + "{p2}",
            ["--uid", "123457"] + self._compose_param_list(dict(p1=5, p2="aaa")),
            "test_main_run_args",
        )
        print(out)
        assert out.find("state: completed") != -1, out
        db = mlrun.get_run_db()
        state, log = db.get_log("123457")
        print(log)
        assert str(log).find(", -x, aaa") != -1, "params not detected in argv"

    def test_main_run_args_with_url_placeholder_missing_env(self):
        args = [
            "--name",
            "test_main_run_args_with_url_placeholder_missing_env",
            "--dump",
            "*",
            "--arg1",
            "value1",
            "--arg2",
            "value2",
        ]
        out = self._exec_main(
            "run",
            args,
            raise_on_error=False,
        )
        out_stdout = out.stdout.decode("utf-8")
        print(out)
        assert (
            out_stdout.find(
                "command/url '*' placeholder is not allowed when code is not from env"
            )
            != -1
        ), out

    def test_main_run_args_with_url_placeholder_from_env(self):
        os.environ["MLRUN_EXEC_CODE"] = b64encode(code.encode("utf-8")).decode("utf-8")
        args = [
            "--name",
            "test_main_run_args_with_url_placeholder_from_env",
            "--uid",
            "123456789",
            "--from-env",
            "--dump",
            "*",
            "--arg1",
            "value1",
            "--arg2",
            "value2",
        ]
        self._exec_main(
            "run",
            args,
            raise_on_error=True,
        )
        db = mlrun.get_run_db()
        _run = db.read_run("123456789")
        print(_run)
        assert _run["status"]["results"]["my_args"] == [
            "main.py",
            "--arg1",
            "value1",
            "--arg2",
            "value2",
        ]
        assert _run["status"]["state"] == "completed"

        args = [
            "--name",
            "test_main_run_args_with_url_placeholder_with_origin_file",
            "--uid",
            "987654321",
            "--from-env",
            "--dump",
            "*",
            "--origin-file",
            "my_file.py",
            "--arg3",
            "value3",
        ]
        self._exec_main(
            "run",
            args,
            raise_on_error=True,
        )
        db = mlrun.get_run_db()
        _run = db.read_run("987654321")
        print(_run)
        assert _run["status"]["results"]["my_args"] == [
            "my_file.py",
            "--arg3",
            "value3",
        ]
        assert _run["status"]["state"] == "completed"

    def test_main_with_url_placeholder(self):
        os.environ["MLRUN_EXEC_CODE"] = b64encode(code.encode("utf-8")).decode("utf-8")
        args = [
            "--name",
            "test_main_with_url_placeholder",
            "--uid",
            "123456789",
            "--from-env",
            "*",
        ]
        self._exec_main(
            "run",
            args,
            raise_on_error=True,
        )
        db = mlrun.get_run_db()
        _run = db.read_run("123456789")
        print(_run)
        assert _run["status"]["results"]["my_args"] == ["main.py"]
        assert _run["status"]["state"] == "completed"

    @pytest.mark.parametrize(
        "op,args,raise_on_error,expected_output",
        [
            # bad flag before command
            [
                "run",
                [
                    "--bad-flag",
                    "--name",
                    "test-main-run-basic",
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
                ["--name", "test-main-run-basic", "--bad-flag"],
                False,
                "Error: Invalid value for '[URL]': URL (--bad-flag) cannot start with '-', "
                "ensure the command options are typed correctly. Preferably use '--' to separate options and "
                "arguments e.g. 'mlrun run --option1 --option2 -- [URL] [--arg1|arg1] [--arg2|arg2]'",
            ],
            # bad flag after -- separator
            [
                "run",
                ["--name", "test-main-run-basic", "--", "-notaflag"],
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
                    "test-main-run-basic",
                    "--",
                    f"{examples_path}/training.py",
                    "--some-arg",
                ],
                True,
                "status: completed",
            ],
        ],
    )
    def test_main_run_args_validation(self, op, args, raise_on_error, expected_output):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                "--out-path",
                tmpdir,
            ] + args
            out = self._exec_main(
                op,
                args,
                raise_on_error=raise_on_error,
            )
            if not raise_on_error:
                out = out.stderr.decode("utf-8")

            assert out.find(expected_output) != -1, out

    def test_main_run_args_from_env(self):
        os.environ["MLRUN_EXEC_CODE"] = b64encode(code.encode("utf-8")).decode("utf-8")
        os.environ["MLRUN_EXEC_CONFIG"] = (
            '{"spec":{"parameters":{"x": "bbb"}},'
            '"metadata":{"uid":"123459", "name":"tst", "labels": {"kind": "job"}}}'
        )

        out = self._exec_run(
            "'main.py -x {x}'",
            ["--from-env"],
            "test_main_run_args_from_env",
        )
        db = mlrun.get_run_db()
        run_object = db.read_run("123459")
        print(out)
        assert run_object["status"]["state"] == "completed", out
        assert run_object["status"]["results"]["my_args"] == [
            "main.py",
            "-x",
            "bbb",
        ], "params not detected in argv"

    @pytest.mark.skipif(sys.platform == "win32", reason="skip on windows")
    def test_main_run_nonpy_from_env(self):
        os.environ["MLRUN_EXEC_CODE"] = b64encode(nonpy_code.encode("utf-8")).decode(
            "utf-8"
        )
        os.environ["MLRUN_EXEC_CONFIG"] = (
            '{"spec":{},"metadata":{"uid":"123411", "name":"tst", "labels": {"kind": "job"}}}'
        )

        # --kfp flag will force the logs to print (for the assert)
        out = self._exec_run(
            "bash {codefile} xx",
            ["--from-env", "--mode", "pass", "--kfp"],
            "test_main_run_nonpy_from_env",
        )
        db = mlrun.get_run_db()
        run_object = db.read_run("123411")
        assert run_object["status"]["state"] == "completed", out
        state, log = db.get_log("123411")
        print(state, log)
        assert str(log).find("abc123 xx") != -1, "incorrect output"

    def test_main_run_pass(self):
        out = self._exec_run(
            "python -c print(56)",
            ["--mode", "pass", "--uid", "123458"],
            "test_main_run_pass",
        )
        print(out)
        assert out.find("state: completed") != -1, out
        db = mlrun.get_run_db()
        state, log = db.get_log("123458")
        assert str(log).find("56") != -1, "incorrect output"

    def test_main_run_pass_args(self):
        out = self._exec_run(
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

    def test_main_run_archive(self):
        args = f"--source {examples_path}/archive.zip --handler handler -p p1=1"
        out = self._exec_run("./myfunc.py", args.split(), "test_main_run_archive")
        assert out.find("state: completed") != -1, out

    def test_main_local_source(self):
        args = f"--source {examples_path} --handler my_func"
        with pytest.raises(Exception) as e:
            self._exec_run("./handler.py", args.split(), "test_main_local_source")
        assert (
            f"source ({examples_path}) must be a compressed (tar.gz / zip) file, "
            f"a git repo, a file path or in the project's context (.)" in str(e.value)
        )

    def test_main_run_archive_subdir(self):
        runtime = '{"spec":{"pythonpath":"./subdir"}}'
        args = f"--source {examples_path}/archive.zip -r {runtime}"
        out = self._exec_run(
            "./subdir/func2.py", args.split(), "test_main_run_archive_subdir"
        )
        print(out)
        assert out.find("state: completed") != -1, out

    def test_main_local_project(self):
        mlrun.get_or_create_project("testproject", allow_cross_project=True)
        project_path = str(self.assets_path)
        args = "-f simple -p x=2 --dump"
        out = self._exec_main("run", args.split(), cwd=project_path)
        assert out.find("state: completed") != -1, out
        assert out.find("y: 4") != -1, out  # y = x * 2

    def test_main_local_flag(self):
        fn = mlrun.code_to_function(
            filename=f"{examples_path}/handler.py", kind="job", handler="my_func"
        )
        yaml_path = f"{out_path}/myfunc.yaml"
        fn.export(yaml_path)
        args = f"-f {yaml_path} --local"
        out = self._exec_run("", args.split(), "test_main_local_flag")
        print(out)
        assert out.find("state: completed") != -1, out

    def test_main_run_class(self):
        function_path = str(self.assets_path / "handler.py")

        out = self._exec_run(
            function_path,
            self._compose_param_list(dict(x=8)) + ["--handler", "MyCls::mtd"],
            "test_main_run_class",
        )
        assert out.find("state: completed") != -1, out
        assert out.find("rx: 8") != -1, out

    def test_run_from_module(self):
        args = [
            "--name",
            "test1",
            "--dump",
            "--handler",
            "json.dumps",
            "-p",
            "obj=[6,7]",
        ]
        out = self._exec_main("run", args)
        assert out.find("state: completed") != -1, out
        assert out.find("return: '[6, 7]'") != -1, out

    def test_get_runs_with_tag(self):
        args = ["runs", "-p", "obj=[6,7]", "--tag", "666"]
        out = self._exec_main("get", args)
        assert out.find("Unsupported argument") != -1, out

    def test_main_env_file(self):
        # test run with env vars loaded from a .env file
        function_path = str(self.assets_path / "handler.py")
        envfile = str(self.assets_path / "envfile")

        out = self._exec_run(
            function_path,
            ["--handler", "env_file_test", "--env-file", envfile],
            "test_main_env_file",
        )
        assert out.find("state: completed") != -1, out
        assert out.find("ENV_ARG1: '123'") != -1, out
        assert out.find("kfp_ttl: 12345") != -1, out

    def test_main_run_function_from_another_project(self):
        # test running function from another project and validate that the function is stored in the current project
        project = mlrun.get_or_create_project("first-project", allow_cross_project=True)

        fn = mlrun.code_to_function(
            name="new-func",
            filename=f"{examples_path}/handler.py",
            kind="local",
            handler="my_func",
        )
        project.set_function(fn)
        fn.save()

        # create another project
        project2 = mlrun.get_or_create_project(
            "second-project", allow_cross_project=True
        )

        # from the second project - run the function that we stored in the first project
        args = (
            "-f db://first-project/new-func --project second-project --ensure-project"
        )
        self._exec_main("run", args.split())

        # validate that the function is now stored also in the second project
        first_project_func = project.get_function("new-func", ignore_cache=True)
        second_project_func = project2.get_function("new-func", ignore_cache=True)

        assert second_project_func is not None
        assert second_project_func.metadata.name == first_project_func.metadata.name
        assert second_project_func.metadata.tag == first_project_func.metadata.tag
        assert second_project_func.metadata.hash != first_project_func.metadata.hash

    @staticmethod
    def _exec_main(op, args, cwd=examples_path, raise_on_error=True):
        cmd = [executable, "-m", "mlrun", op]
        if args:
            cmd += args
        out = run(cmd, capture_output=True, cwd=cwd)
        if out.returncode != 0:
            print(out.stderr.decode("utf-8"), file=stderr)
            print(out.stdout.decode("utf-8"), file=stderr)
            print(traceback.format_exc())
            if raise_on_error:
                raise Exception(out.stderr.decode("utf-8"))
            else:
                # return out so that we can check the error message on stdout and stderr
                return out

        return out.stdout.decode("utf-8")

    def _exec_run(self, cmd, args, test, raise_on_error=True):
        args = args + ["--name", test, "--dump", cmd]
        return self._exec_main("run", args, raise_on_error=raise_on_error)

    @staticmethod
    def _compose_param_list(params: dict, flag="-p"):
        composed_params = []
        for k, v in params.items():
            composed_params += [flag, f"{k}={v}"]
        return composed_params
