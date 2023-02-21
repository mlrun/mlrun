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
import datetime
import os
from sys import executable

import pytest

import mlrun
import tests.system.base


def exec_run(args):
    cmd = [executable, "-m", "mlrun", "run"] + args
    out = os.popen(" ".join(cmd)).read()
    return out


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
            requirements=["pandas"],
        )
        assert function.spec.image == ""
        assert function.spec.build.base_image == expected_base_image
        function.deploy()
        assert function.spec.image == expected_spec_image
        function.run()

    def test_store_function_after_run_local_verify_credentials_are_masked(self):
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

        proj = mlrun.get_or_create_project(self.project_name, self.results_path)
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
        mlrun.get_or_create_project(self.project_name, self.results_path)

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
            "/usr/local/bin/mlrun",
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
        mlrun.get_or_create_project(self.project_name, self.results_path)

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
        project = mlrun.get_or_create_project(self.project_name, self.results_path)
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
            run = function.run(handler="mycls::mtd", params=params)
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
        exec_run(args)
        end_time = datetime.datetime.now()

        assert (
            end_time - start_time
        ).seconds >= time_to_sleep, "run did not wait for completion"

        runs = mlrun.get_run_db().list_runs(project=self.project_name, name=run_name)
        assert len(runs) == 1

    def test_function_handler_set_labels_and_annotations(self):
        code_path = str(self.assets_path / "handler.py")
        mlrun.get_or_create_project(self.project_name, self.results_path)

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
