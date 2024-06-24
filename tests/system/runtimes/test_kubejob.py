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
import datetime
import json
import subprocess
from sys import executable

import pandas as pd
import pytest

import mlrun
import mlrun.common.schemas
import mlrun.feature_store.common
import mlrun.model
import tests.system.base
from mlrun.runtimes.function_reference import FunctionReference


def exec_cli(args, action="run"):
    cmd = [executable, "-m", "mlrun", action] + args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    ret_code = process.returncode
    return out.decode(), err.decode(), ret_code


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestKubejobRuntime(tests.system.base.TestMLRunSystem):
    project_name = "kubejob-system-test"

    def test_deploy_function(self):
        code_path = str(self.assets_path / "kubejob_function.py")

        function = mlrun.code_to_function(
            name="simple-function",
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.build_config(base_image="mlrun/mlrun", commands=["pip install pandas"])

        self._logger.debug("Deploying kubejob function")
        function.deploy()

    def test_deploy_function_with_requirements_file(self):
        # ML-3518
        code_path = str(self.assets_path / "kubejob_function_custom_requirements.py")
        requirements_path = str(self.assets_path / "requirements-test.txt")
        function = mlrun.code_to_function(
            name="simple-function",
            kind="job",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
            requirements_file=requirements_path,
        )
        function.deploy()
        run = function.run(handler="MyCls::do")
        outputs = run.outputs
        assert "requests" in outputs, "requests not in outputs"
        assert "chardet" in outputs, "chardet not in outputs"
        assert "pyhive" in outputs, "pyhive not in outputs"

    def test_deploy_function_without_image_with_requirements(self):
        # ML-2669
        code_path = str(self.assets_path / "kubejob_function.py")
        expected_spec_image = ".mlrun/func-kubejob-system-test-simple-function:latest"
        expected_base_image = "mlrun/mlrun"

        function = mlrun.code_to_function(
            name="simple-function",
            kind="job",
            project=self.project_name,
            filename=code_path,
            requirements=[
                # ML-3518
                "pandas>=1.5.0, <3",
            ],
        )
        assert function.spec.image == ""
        assert function.spec.build.base_image == expected_base_image
        function.deploy()
        assert function.spec.image == expected_spec_image
        function.run()

    def test_store_function_is_not_failing_if_generate_access_key_not_requested(self):
        code_path = str(self.assets_path / "kubejob_function.py")
        function_name = "simple-function"
        function = mlrun.code_to_function(
            name=function_name,
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        hash_key = function.save(versioned=True)
        function = mlrun.get_run_db().get_function(
            function_name, project=self.project_name, tag="latest", hash_key=hash_key
        )
        assert not function["metadata"].get("credentials", {}).get("access_key", None)

    @pytest.mark.enterprise
    def test_store_function_after_run_local_verify_credentials_are_masked(self):
        """
        This test is verifying that when running a function locally and then storing it with requesting to generate
        access key, the credentials are masked.
        Skip on CE because we don't have auth in CE and therefore there are no credentials to mask.
        """
        code_path = str(self.assets_path / "kubejob_function.py")
        function_name = "simple-function"
        function = mlrun.code_to_function(
            name=function_name,
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.run(local=True)
        assert function.metadata.credentials.access_key.startswith(
            mlrun.model.Credentials.generate_access_key
        )

        hash_key = mlrun.get_run_db().store_function(
            function.to_dict(), function_name, self.project_name
        )
        masked_function = mlrun.get_run_db().get_function(
            function.metadata.name, self.project_name, tag="latest", hash_key=hash_key
        )
        masked_function_obj = mlrun.new_function(runtime=masked_function)
        assert masked_function_obj.metadata.credentials.access_key.startswith(
            mlrun.model.Credentials.secret_reference_prefix
        )
        # TODO: once env is sanitized attribute no need to use the camelCase anymore and rather access it is k8s class
        assert (
            masked_function_obj.get_env("V3IO_ACCESS_KEY")["secretKeyRef"] is not None
        )

    def test_deploy_function_after_deploy(self):
        # ML-2701
        code_path = str(self.assets_path / "kubejob_function.py")
        expected_spec_image = ".mlrun/func-kubejob-system-test-simple-function:latest"
        expected_base_image = "mlrun/mlrun"
        function = mlrun.code_to_function(
            "simple-function",
            kind="job",
            image="mlrun/mlrun",
            filename=code_path,
            requirements=["pandas"],
        )
        assert function.spec.build.base_image == expected_base_image
        assert function.spec.image == ""

        function.deploy()
        assert function.spec.image == expected_spec_image
        assert function.spec.build.base_image == expected_base_image

        function.deploy()
        assert function.spec.image == expected_spec_image
        assert function.spec.build.base_image == expected_base_image

    def test_function_with_param(self):
        code_path = str(self.assets_path / "function_with_params.py")

        proj = mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )
        project_param = "some value"
        local_param = "my local param"
        proj.spec.params = {"project_param": project_param}
        proj.save()

        function = mlrun.code_to_function(
            name="function-with-params",
            kind="job",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        run = function.run(params={"param1": local_param})
        assert run.status.results["project_param"] == project_param
        assert run.status.results["param1"] == local_param

    def test_function_handler_with_args(self):
        code_path = str(self.assets_path / "function_with_args.py")
        mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )

        function = mlrun.code_to_function(
            name="function-with-args",
            kind="job",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        args = ["--some-arg", "a-value-123"]
        function.spec.args = args
        run = function.run()
        assert run.status.results["some-arg-by-handler"] == args[1]
        assert run.status.results["my-args"] == [
            "/opt/conda/bin/mlrun",
            "run",
            "--name",
            "function-with-args-handler",
            "--from-env",
            "--handler",
            "handler",
            "--origin-file",
            code_path,
            "*",
            "--some-arg",
            "a-value-123",
        ]

    def test_function_with_args(self):
        code_path = str(self.assets_path / "function_with_args.py")
        mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )

        function = mlrun.code_to_function(
            name="function-with-args",
            kind="job",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        args = ["--some-arg", "a-value-123"]
        function.spec.args = args
        run = function.run()
        assert run.status.results["some-arg-by-main"] == args[1]
        assert run.status.results["my-args"] == [
            "function_with_args.py",
            "--some-arg",
            "a-value-123",
        ]

    @pytest.mark.enterprise
    def test_new_function_with_args(self):
        """
        skip this test on ce because it requires uploading artifacts to target store
        we don't allow uploading to s3 from tests and we only allow downloading compressed files from remote sources
        here we upload the python code file to v3io
        """
        code_path = str(self.assets_path / "function_with_args.py")
        project = mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )
        art = project.log_artifact(
            "my_code_artifact", local_path=code_path, format="py"
        )

        function = mlrun.new_function(
            name="new-function-with-args",
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
            source=art.get_target_path(),
            command="my_code_artifact.py --another-one 123",
        )

        args = ["--some-arg", "val-with-artifact"]
        function.spec.args = args
        function.deploy()
        run = function.run()
        assert run.status.results["some-arg-by-main"] == args[1]
        assert run.status.results["another-one"] == "123"
        assert run.status.results["my-args"] == [
            "my_code_artifact.py",
            "--another-one",
            "123",
            "--some-arg",
            "val-with-artifact",
        ]

    @pytest.mark.parametrize("local", [True, False])
    def test_log_artifact_with_run_function(self, local):
        train_path = str(self.assets_path / "log_artifact.py")
        function_parameter = 100
        self.project.set_function(
            train_path,
            name="log-artifact",
            image="mlrun/mlrun",
            kind="job",
            handler="train",
        )
        self.project.run_function(
            "log-artifact", params={"i": function_parameter}, local=local
        )
        resource = self.project.get_store_resource(
            f"store://datasets/{self.project_name}/log-artifact-train_df#0:latest"
        ).to_dataitem()
        expected_df = pd.DataFrame(
            {f"col{function_parameter}": [function_parameter] * 10}
        )
        result_df = resource.as_df()
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_function_with_kwargs(self):
        code_path = str(self.assets_path / "function_with_kwargs.py")
        mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )

        function = mlrun.code_to_function(
            name="function-with-kwargs",
            kind="job",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        kwargs = {"some_arg": "a-value-123", "another_arg": "another-value-456"}
        params = {"x": "2"}
        params.update(kwargs)
        run = function.run(params=params, handler="func")
        assert run.outputs["return"] == kwargs

    def test_class_handler(self):
        code_path = str(self.assets_path / "kubejob_function.py")
        cases = [
            ({"y": 3}, {"rx": 0, "ry": 3, "ra1": 1}),
            ({"_init_args": {"a1": 9}, "y": 5}, {"rx": 0, "ry": 5, "ra1": 9}),
        ]
        function = mlrun.code_to_function(
            "function-with-class",
            filename=code_path,
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        for params, results in cases:
            run = function.run(handler="MyCls::mtd", params=params)
            print(run.to_yaml())
            assert run.status.results == results

    def test_run_from_module(self):
        function = mlrun.new_function(
            "function-from-module",
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        run = function.run(handler="json.dumps", params={"obj": {"x": 99}})
        print(run.status.results)
        assert run.output("return") == '{"x": 99}'

    def test_run_cli_watch_remote_job(self):
        sleep_func = mlrun.code_to_function(
            "sleep-function",
            filename=str(self.assets_path / "sleep.py"),
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        self.project.set_function(sleep_func)
        self.project.sync_functions(save=True)

        run_name = "watch-test"
        # ideally we wouldn't add sleep to a test, but in this scenario where we want to make sure that we actually
        # wait for the run to finish, and because we can't be sure how long it will take to spawn the pod and run the
        # function, we need to set pretty long timeout
        time_to_sleep = 30
        args = [
            "--name",
            run_name,
            "--func-url",
            f"db://{self.project_name}/sleep-function",
            "--watch",
            "--project",
            self.project_name,
            "--param",
            f"time_to_sleep={time_to_sleep}",
            "--handler",
            "handler",
        ]
        start_time = datetime.datetime.now()
        exec_cli(args)
        end_time = datetime.datetime.now()

        assert (
            end_time - start_time
        ).seconds >= time_to_sleep, "run did not wait for completion"

        runs = mlrun.get_run_db().list_runs(project=self.project_name, name=run_name)
        assert len(runs) == 1

    def test_run_cli_not_specified_image(self):
        # define the function without an image
        func = mlrun.code_to_function(
            "new-function",
            filename=str(self.assets_path / "my_func.py"),
            kind="job",
            project=self.project_name,
        )
        self.project.set_function(func)
        self.project.sync_functions(save=True)

        # when image is not provided, "mlrun/mlrun" should be used by default
        args = [
            "--func-url",
            f"db://{self.project_name}/new-function",
            "my_func.py",
            "--project",
            self.project_name,
            "--handler",
            "handler",
        ]
        _, _, ret_code = exec_cli(args)
        assert ret_code == 0

    def test_cli_build_function_without_kind(self):
        # kind='job' should be used by default, the user is not required to specify it
        function = str(self.assets_path / "function_without_kind.yaml")
        args = [
            "--name",
            "test",
            function,
        ]
        out, _, _ = exec_cli(args, action="build")
        assert "Function built, state=ready" in out

    def test_cli_build_runtime_without_kind(self):
        # kind='job' should be used by default, the user is not required to specify it
        # send runtime spec without kind
        runtime = {"metadata": {"name": "test-func"}}
        args = [
            "--name",
            "test",
            "--runtime",
            json.dumps(runtime),
        ]
        out, _, _ = exec_cli(args, action="build")
        assert "Function built, state=ready" in out

    @pytest.mark.parametrize("local", [True, False])
    def test_df_as_params(self, local):
        df = pd.read_parquet(str(self.assets_path / "test_data.parquet"))
        code = """
def print_df(df):
    print(df)
"""
        function_ref = FunctionReference(
            kind="job",
            code=code,
            image="mlrun/mlrun",
            name="test_df_as_param",
        )

        function = function_ref.to_function()
        if local:
            function.run(handler="print_df", params={"df": df}, local=True)
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError) as error:
                function.run(handler="print_df", params={"df": df}, local=False)
            assert (
                "Parameter 'df' has an unsupported value of type 'pandas.DataFrame'"
                in str(error.value)
            )

    def test_function_handler_set_labels_and_annotations(self):
        code_path = str(self.assets_path / "handler.py")
        mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )

        function = mlrun.code_to_function(
            name="test-func",
            kind="job",
            handler="set_labels_and_annotations_handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        run = function.run()
        assert run.metadata.labels.get("label1") == "label-value1"
        assert run.metadata.annotations.get("annotation1") == "annotation-value1"

    def test_normalize_run_name(self):
        function = mlrun.feature_store.common.RunConfig().to_function(
            default_kind="job",
            default_image="mlrun/mlrun",
        )
        function.with_code(str(self.assets_path / "handler.py"))

        task = mlrun.model.new_task(
            name="ASC_merger", handler="set_labels_and_annotations_handler"
        )
        run = function.run(task, project=self.project_name)

        # Before the change of ML-3265 this test should've fail because no normalization was applied on the task name
        assert run.metadata.name == "asc-merger"

    def test_function_with_builder_env(self):
        name = "test-build-env-vars"
        builder_env_key = "ARG1"
        builder_env_val = "value1"

        extra_args_env_key = "ARG2"
        extra_args_env_val = "value2"
        extra_args_flag = "--skip-tls-verify"
        expected_results = [builder_env_val, extra_args_env_val]

        extra_args = (
            f"--build-arg {extra_args_env_key}={extra_args_env_val} {extra_args_flag}"
        )
        code_path = str(self.assets_path / "function_with_env_vars.py")
        project = mlrun.get_or_create_project(
            self.project_name, self.results_path, allow_cross_project=True
        )

        image_name = ".test-custom-image"
        project.build_image(
            image=image_name,
            set_as_default=True,
            with_mlrun=False,
            base_image="mlrun/mlrun",
            requirements=["vaderSentiment"],
            commands=[
                f"echo ${builder_env_key} > /tmp/args.txt",
                f"echo ${extra_args_env_key} >> /tmp/args.txt",
            ],
            builder_env={builder_env_key: builder_env_val},
            extra_args=extra_args,
        )
        project.set_function(
            code_path,
            name=name,
            image=image_name,
            kind="job",
            handler="handler",
        )

        run = project.run_function(name)
        results = run.status.results["results"]
        assert results == expected_results

    def test_abort_run(self):
        sleep_func = mlrun.code_to_function(
            "sleep-function",
            filename=str(self.assets_path / "sleep.py"),
            kind="job",
            project=self.project_name,
            image="mlrun/mlrun",
        )
        run = sleep_func.run(
            params={"time_to_sleep": 30},
            watch=False,
        )
        db = mlrun.get_run_db()
        background_task = db.abort_run(run.metadata.uid)
        assert (
            background_task.status.state
            == mlrun.common.schemas.BackgroundTaskState.succeeded
        )

        run = db.read_run(run.metadata.uid)
        assert (
            run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.aborted
        )

        # list background tasks
        background_tasks = db.list_project_background_tasks()
        assert background_task.metadata.name in [
            task.metadata.name for task in background_tasks
        ]
