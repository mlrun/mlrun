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
import tempfile
import time

import kfp.dsl as dsl
import pytest

import mlrun
import mlrun.datastore.datastore_profile as datastore_profile
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

            assert str(p1) == "{{pipelineparam:op=;name=p1}}", (
                f"p1 was expected to be a pipeline param but is {p1}"
            )

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

        assert run["run"].get("error") == "main: Error (exit code 1)"

    # TODO - uncomment when system tests is bumped to kfp 2.0+ (IGZ 3.7+)
    @pytest.mark.skip(reason="Not supported in kfp<2.0")
    def test_kfp_terminate_pipeline(self):
        code_path = str(self.assets_path / "sleep.py")
        self.project.set_function(
            func=code_path,
            name="sleep-func",
            kind="job",
            image="mlrun/mlrun",
            handler="handler",
        )

        # 1. define a pipeline that sleeps for a few seconds
        @dsl.pipeline(name="terminate-test", description="pipeline to test termination")
        def terminate_pipeline(time_to_sleep: int = 10):
            mlrun.run_function("sleep-func", params={"time_to_sleep": time_to_sleep})

        # 2. Start the pipeline run
        run_id = self.project.run(
            workflow_handler=terminate_pipeline,
            engine="kfp",
            arguments={"time_to_sleep": 60},
            name="terminate-exp",
            watch=False,
        )

        # 3. Wait for it to start
        while True:
            db = mlrun.get_run_db()
            record = db.get_pipeline(run_id, project=self.project_name)
            if record["run"].get("status") == RunStatuses.running:
                break
            time.sleep(1)

        # 4. issue a termination request
        mlrun.terminate_pipeline(run_id, project=self.project_name)

        # 5. wait for it to finish, expecting failed status
        mlrun.wait_for_pipeline_completion(
            run_id,
            project=self.project_name,
            expected_statuses=[RunStatuses.failed],
        )

        # 6. verify the run record shows a termination error
        db = mlrun.get_run_db()
        record = db.get_pipeline(run_id, project=self.project_name)
        err = record["run"].get("status", "")
        assert "failed" in err.lower(), f"expected failed error, got: {err}"

    def test_kfp_long_pipeline_name_steps_have_run_uid(self):
        """Verify that pipeline steps are correctly matched to MLRun runs
        even when the pipeline + step names produce pod hostnames exceeding
        the 63-char kubelet truncation limit."""
        code_path = str(self.assets_path / "my_func.py")
        func = mlrun.code_to_function(
            name="func",
            kind="job",
            filename=code_path,
            project=self.project_name,
            image="mlrun/mlrun",
        )
        self.project.set_function(func)

        # Use a long pipeline name so that pipeline-name + step-name + node-id > 63 chars
        long_pipeline_name = "long-pipeline-name-with-many-characters"

        @dsl.pipeline(name=long_pipeline_name)
        def long_name_pipeline():
            mlrun.run_function(
                "func",
                name="data-preprocessing-and-validation",
                handler="handler",
                params={"p1": 1},
            )
            mlrun.run_function(
                "func",
                name="model-training-and-evaluation",
                handler="handler",
                params={"p1": 2},
            )

        run_id = self.project.run(
            workflow_handler=long_name_pipeline,
            name="long-name-test",
            watch=True,
        )

        mlrun.wait_for_pipeline_completion(run_id, project=self.project_name)

        db = mlrun.get_run_db()
        pipeline = db.get_pipeline(run_id, project=self.project_name)
        graph = pipeline.get("graph", {})

        steps_with_run_uid = [
            step_name
            for step_name, step_info in graph.items()
            if step_info.get("run_uid")
        ]
        assert len(steps_with_run_uid) == 2, (
            f"Expected 2 steps with run_uid, got {len(steps_with_run_uid)}. "
            f"Graph: {graph}"
        )

    @pytest.mark.parametrize("engine", ["kfp", "remote"])
    def test_run_workflow_from_store_artifact(self, engine):
        """Run a workflow whose ``workflow_path`` is a ``store://`` CodeArtifact
        URI, end-to-end against the cluster (ML-11981)."""
        # The remote engine requires `project.spec.source` to be a cloneable
        # URL (for the runner pod's `load_project` call). Auto-detection from
        # the worktree picks up `git@...` which the server rejects, so set
        # a known-cloneable demo repo. Its content isn't used here — the
        # function comes from hub://describe and the workflow from store://.
        self.project.spec.source = "git://github.com/mlrun/project-demo.git"
        self.project.save()
        self.project.set_function("hub://describe", "describe")

        workflow_src = self._write_kfp_pipeline_tempfile()
        try:
            artifact = self.project.log_code_file(
                key=f"{engine}_workflow_code",
                local_path=workflow_src,
                code_type="workflow",
            )
            store_uri = artifact.uri

            workflow_name = f"store_pipeline_{engine}"
            self.project.set_workflow(
                workflow_name, workflow_path=store_uri, engine=engine
            )
            run = self.project.run(
                workflow_name,
                watch=True,
                engine=engine,
                artifact_path=f"v3io:///projects/{self.project_name}",
            )
            assert run.state == RunStatuses.succeeded, (
                f"workflow did not finish successfully (state={run.state})"
            )
        finally:
            try:
                os.unlink(workflow_src)
            except OSError:
                pass

    @pytest.mark.parametrize("engine", ["kfp", "remote"])
    def test_run_workflow_with_ds_profile_target(self, engine):
        """The workflow CodeArtifact's target_path is behind a ds:// v3io
        profile. Exercises the secrets / profile-resolution path inside
        load_source_code → get_dataitem at the get_source_file call site
        (client-side for engine=kfp, runner-pod-side for engine=remote)."""
        access_key = os.environ.get("V3IO_ACCESS_KEY")
        assert access_key, "V3IO_ACCESS_KEY required for this system test"

        profile_name = "wf-store-test-v3io-profile"
        profile = datastore_profile.DatastoreProfileV3io(
            name=profile_name, v3io_access_key=access_key
        )
        self.project.register_datastore_profile(profile)
        datastore_profile.register_temporary_client_datastore_profile(profile)

        # Cloneable project source for the remote runner pod (its content is
        # unused — function is hub://, workflow is store://).
        self.project.spec.source = "git://github.com/mlrun/project-demo.git"
        self.project.save()

        workflow_src = self._write_kfp_pipeline_tempfile()
        try:
            artifact_key = f"ds_workflow_code_{engine}"
            artifact = self.project.log_code_file(
                key=artifact_key,
                local_path=workflow_src,
                code_type="workflow",
                target_path=(
                    f"ds://{profile_name}/projects/{self.project_name}/code/"
                    f"{artifact_key}.py"
                ),
            )
            store_uri = artifact.uri

            self.project.set_function("hub://describe", "describe")
            workflow_name = f"store_pipeline_ds_{engine}"
            self.project.set_workflow(
                workflow_name, workflow_path=store_uri, engine=engine
            )
            run = self.project.run(
                workflow_name,
                watch=True,
                engine=engine,
                artifact_path=f"v3io:///projects/{self.project_name}",
            )
            assert run.state == RunStatuses.succeeded, (
                f"workflow did not finish successfully (state={run.state})"
            )
        finally:
            try:
                os.unlink(workflow_src)
            except OSError:
                pass

    @staticmethod
    def _write_kfp_pipeline_tempfile() -> str:
        """Write a minimal kfp pipeline source to a temp file and return its path.
        Caller is responsible for cleanup."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as wf:
            wf.write(
                "from kfp import dsl\n"
                "import mlrun\n"
                "\n"
                "funcs = {}\n"
                "\n"
                '@dsl.pipeline(name="store-uri pipeline")\n'
                "def kfpipeline():\n"
                '    funcs["describe"].as_step(name="describe-step")\n'
            )
            return wf.name
