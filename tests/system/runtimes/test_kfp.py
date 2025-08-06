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

import os
import time

import pytest
from kfp import dsl

import mlrun
import mlrun.runtimes.mounts
import tests.system.base
from mlrun import mlconf
from mlrun_pipelines.common.models import RunStatuses


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestKFP(tests.system.base.TestMLRunSystem):
    project_name = "kfp-system-test"

    @pytest.mark.enterprise
    def test_kfp_with_mount(self):
        code_path = str(self.assets_path / "kfp_with_mount.py")
        kfp_with_v3io_mount = mlrun.code_to_function(
            name="my-kfp-with-mount",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )
        kfp_with_v3io_mount.apply(mlrun.runtimes.mounts.mount_v3io())

        @dsl.pipeline(name="job test", description="demonstrating mlrun usage")
        def job_pipeline(p1=9):
            kfp_with_v3io_mount.as_step(
                handler="handler", params={"p1": p1}, outputs=["mymodel"]
            )

        out = mlconf.artifact_path or os.path.abspath("./data")
        artifact_path = os.path.join(out, "{{run.uid}}")
        arguments = {"p1": 8}
        run_id = mlrun._run_pipeline(
            job_pipeline,
            arguments,
            experiment="my-job",
            artifact_path=artifact_path,
            project=self.project_name,
        )

        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)

    @pytest.mark.enterprise
    def test_kfp_with_pipeline_param_and_run_function(self):
        code_path = str(self.assets_path / "my_func.py")
        func = mlrun.code_to_function(
            name="func",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )
        self.project.set_function(func)

        @dsl.pipeline(name="job test", description="demonstrating mlrun usage")
        def job_pipeline(p1=9):
            mlrun.run_function(
                "func",
                handler="handler",
                params={"p1": p1},
                outputs=["mymodel"],
            )

            assert (
                str(p1) == "{{pipelineparam:op=;name=p1}}"
            ), f"p1 was expected to be a pipeline param but is {p1}"

        arguments = {"p1": 8}
        run_id = self.project.run(
            workflow_handler=job_pipeline,
            arguments=arguments,
            name="my-job",
            watch=True,
        )

        # double check that the pipeline completed successfully
        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)

    def test_kfp_without_image(self):
        code_path = str(self.assets_path / "my_func.py")
        my_func = mlrun.code_to_function(
            name="my-kfp-without-image",
            kind="job",
            filename=code_path,
            project=self.project_name,
        )

        @dsl.pipeline(name="job no image test", description="kfp without image test")
        def job_pipeline():
            my_func.as_step(handler="handler", auto_build=True)

        run_id = mlrun._run_pipeline(
            job_pipeline,
            experiment="my-job",
            project=self.project_name,
        )

        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)

    def test_kfp_retry(self):
        code_path = str(self.assets_path / "my_func.py")
        my_func = mlrun.code_to_function(
            name="my-kfp-without-image",
            kind="job",
            filename=code_path,
            project=self.project_name,
        )

        @dsl.pipeline(name="job no image test", description="kfp without image test")
        def job_pipeline():
            my_func.as_step(handler="handler", auto_build=True)

        run_id = mlrun._run_pipeline(
            job_pipeline,
            experiment="my-job",
            project=self.project_name,
        )

        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)
        new_run_id = mlrun.retry_pipeline(run_id, project=self.project_name)
        mlrun.wait_for_pipeline_completion(new_run_id, project=self.project_name)
        assert (
            new_run_id != run_id
        )  # On successful runs, a new ID is generated because the pipeline is cloned

    @pytest.mark.enterprise
    def test_kfp_with_failed_pipeline(self):
        code_path = str(self.assets_path / "raise_func.py")
        func = mlrun.code_to_function(
            name="func",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )
        self.project.set_function(func)

        @dsl.pipeline(name="job test", description="demonstrating mlrun usage")
        def job_pipeline():
            mlrun.run_function(
                "func",
                handler="handler",
                outputs=["mymodel"],
            )

        run_id = self.project.run(
            workflow_handler=job_pipeline,
            name="my-job",
        )

        # double check that the pipeline completed successfully
        mlrun.wait_for_pipeline_completion(
            run_id, project=self.project_name, expected_statuses=[RunStatuses.failed]
        )
        db = mlrun.get_run_db()
        run = db.get_pipeline(run_id, project=self.project_name)

        assert run["run"].get("error") == "Error (exit code 1)"

    def test_kfp_terminate_pipeline(self):
        code_path = str(self.assets_path / "sleep.py")
        sleep_fn = mlrun.code_to_function(
            name="sleep-func",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )

        # 1. define a pipeline that sleeps for a few seconds
        @dsl.pipeline(name="terminate-test", description="pipeline to test termination")
        def terminate_pipeline(time_to_sleep: int = 10):
            sleep_fn.as_step(
                handler="handler",
                params={"time_to_sleep": time_to_sleep},
            )

        # 2. launch it (non-blocking)
        run_id = mlrun._run_pipeline(
            terminate_pipeline,
            arguments={"time_to_sleep": 60},
            experiment="terminate-exp",
            project=self.project_name,
        )

        # 3. Wait for it to start
        while True:
            db = mlrun.get_run_db()
            record = db.get_pipeline(run_id, project=self.project_name)
            if record["run"].get("status") == RunStatuses.running:
                break
            time.sleep(1)

        # 4. issue a termination request
        terminate_task_id = mlrun.terminate_pipeline(run_id, project=self.project_name)

        time.sleep(10)  # wait a bit to ensure the termination request is processed
        duplicate_terminate_task_id = mlrun.terminate_pipeline(
            run_id, project=self.project_name
        )

        assert (
            terminate_task_id == duplicate_terminate_task_id
        ), "Duplicate termination requests should return the same task ID"

        # 5. wait for it to finish, expecting failed status
        mlrun.wait_for_pipeline_completion(
            run_id,
            project=self.project_name,
            expected_statuses=[RunStatuses.failed],
        )

        # 6. verify the run record shows a termination error
        db = mlrun.get_run_db()
        record = db.get_pipeline(run_id, project=self.project_name)
        err = record["run"].get("error", "")
        assert "failed" in err.lower(), f"expected failed error, got: {err}"
