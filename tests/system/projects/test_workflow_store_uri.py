# Copyright 2026 Iguazio
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

import pytest

import mlrun.datastore.datastore_profile as datastore_profile
import mlrun_pipelines.common.models
import tests.system.base

_PIPELINE_BODY = (
    "from kfp import dsl\n"
    "import mlrun\n"
    "\n"
    "funcs = {}\n"
    "\n"
    '@dsl.pipeline(name="store-uri pipeline")\n'
    "def kfpipeline():\n"
    '    funcs["describe"].as_step(name="describe-step")\n'
)


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestWorkflowStoreUri(tests.system.base.TestMLRunSystem):
    """End-to-end coverage for set_workflow with store:// CodeArtifact URIs.

    Validates the workflow store:// resolution path added by ML-11981:

      1. log_code_file(code_type="workflow") registers a workflow CodeArtifact.
      2. set_workflow(workflow_path="store://...") accepts the URI and
         validates code_type client-side.
      3. project.run() triggers _KFPRunner.save / _get_handler (engine="kfp")
         OR submit_workflow → workflow runner pod (engine="remote"); both
         call WorkflowSpec.get_source_file with the new project_name argument;
         the store:// branch downloads the workflow source via
         mlrun.utils.clones.load_source_code and KFP compiles + executes the
         pipeline against the cluster.

    Uses ``hub://describe`` for the pipeline step so no local function
    source / project source is needed — the runner just resolves the
    workflow file, compiles, and submits to KFP.
    """

    project_name = "workflow-store-uri-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._tmp_workflow_files: list[str] = []

    def custom_teardown(self):
        super().custom_teardown()
        for path in self._tmp_workflow_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _write_pipeline_to_tempfile(self) -> str:
        """Write the kfp pipeline source to a temp file and track it for teardown."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as wf:
            wf.write(_PIPELINE_BODY)
            workflow_src = wf.name
        self._tmp_workflow_files.append(workflow_src)
        return workflow_src

    def _setup_store_workflow(self, artifact_key: str) -> str:
        """Log the workflow source as a CodeArtifact and return its store:// URI."""
        workflow_src = self._write_pipeline_to_tempfile()
        artifact = self.project.log_code_file(
            key=artifact_key,
            local_path=workflow_src,
            code_type="workflow",
        )
        return artifact.uri

    @pytest.mark.parametrize("engine", ["kfp", "remote"])
    def test_run_workflow_from_store_artifact(self, engine):
        """Run a workflow whose ``workflow_path`` is a ``store://`` CodeArtifact
        URI, end-to-end against the cluster."""
        # The remote engine requires `project.spec.source` to be a cloneable
        # URL (for the runner pod's `load_project` call). Auto-detection from
        # the worktree picks up `git@...` which the server rejects, so set
        # a known-cloneable demo repo. Its content isn't used here — the
        # function comes from hub://describe and the workflow from store://.
        self.project.spec.source = "git://github.com/mlrun/project-demo.git"
        self.project.save()
        self.project.set_function("hub://describe", "describe")
        store_uri = self._setup_store_workflow(f"{engine}_workflow_code")

        workflow_name = f"store_pipeline_{engine}"
        self.project.set_workflow(workflow_name, workflow_path=store_uri, engine=engine)
        run = self.project.run(
            workflow_name,
            watch=True,
            engine=engine,
            artifact_path=f"v3io:///projects/{self.project_name}",
        )

        assert run.state == mlrun_pipelines.common.models.RunStatuses.succeeded, (
            f"workflow did not finish successfully (state={run.state})"
        )

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

        artifact_key = f"ds_workflow_code_{engine}"
        workflow_src = self._write_pipeline_to_tempfile()
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

        assert run.state == mlrun_pipelines.common.models.RunStatuses.succeeded, (
            f"workflow did not finish successfully (state={run.state})"
        )
