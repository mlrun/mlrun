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

        print(mlrun.get_run_db().list_functions(project=self.project_name))

        run_name = "watch-test"
        # ideally we wouldn't add sleep to a test, but in this scenario where we want to make sure that we actually
        # wait for the run to finish, and because we can't be sure how long it will take to spawn the pod and run the
        # function, we need to set pretty long timeout
        time_to_sleep = 30
        # exec the workflow and set a short timeout, should fail
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
