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

import mlrun.common.constants
import mlrun.datastore.datastore_profile as datastore_profile
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioStoreUri(tests.system.base.TestMLRunSystem):
    """End-to-end coverage for vanilla Nuclio with store:// CodeArtifact sources.

    These tests validate the init-container path the server installs when a
    Nuclio function's source is a store:// CodeArtifact URI:

      1. Source URI is stashed to status.application_source so the Nuclio
         builder doesn't see store:// (which it cannot resolve).
      2. An init container runs `mlrun load-source` and writes code into a
         shared volume.
      3. The main function container picks up the code via PYTHONPATH.
    """

    project_name = "nuclio-store-uri-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._handler_filename = "echo_handler.py"
        self._function_handler = "echo_handler:handler"
        self._function_image = os.environ.get("MLRUN_TEST_IMAGE")

    def _log_code_artifact(self, key: str, src_path: str | None = None) -> str:
        """Log a CodeArtifact and return its canonical store:// URI.

        :param key:      Artifact key (must be unique per test).
        :param src_path: Path to the source file. Defaults to the test's
                         echo_handler asset.
        :returns: The artifact's own ``.uri`` (whatever the system stored,
                  including any tag suffix). Avoids reconstructing the URI
                  from a string template — see TESTING_STANDARDS.md §7.
        """
        local_path = src_path or os.path.join(self.assets_path, self._handler_filename)
        artifact = self.project.log_code_file(key=key, local_path=local_path)
        return artifact.uri

    def _assert_init_container_present(self, function):
        """Assert the source-loader init container is wired correctly."""
        init_containers = function.spec.config.get("spec.initContainers") or []
        loader_containers = [
            c
            for c in init_containers
            if c.get("name") == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
        ]
        assert len(loader_containers) == 1, (
            f"Expected exactly one source-loader init container, got "
            f"{len(loader_containers)} (all init containers: {init_containers})"
        )
        assert loader_containers[0]["command"] == ["mlrun", "load-source"]

    @staticmethod
    def _versioned_handler_body(version: str) -> str:
        """Return a Nuclio handler that responds with its module VERSION."""
        return (
            "import json\n"
            "from http import HTTPStatus\n\n"
            "import nuclio_sdk\n\n"
            f"VERSION = {version!r}\n\n"
            "def handler(context, event):\n"
            "    return context.Response(\n"
            "        body=json.dumps({'version': VERSION}),\n"
            "        headers={},\n"
            "        content_type='application/json',\n"
            "        status_code=HTTPStatus.OK,\n"
            "    )\n"
        )

    def test_deploy_and_artifact_update(self):
        """End-to-end deploy + artifact-update redeploy via a DataStore profile.

        Combines three concerns into a single deploy/redeploy cycle:

        1. Artifact-update redeploy semantics — the init container fetches the
           *current* artifact target on each deploy, not a cached copy
           (log_code_file v1 → deploy → re-log v2 under same key+tag →
           redeploy → handler returns v2).

        2. Init container envFrom mount — asserts the deployed init container
           references mlrun-project-secrets-<project> via envFrom. Verified on
           the v1 deploy that's already happening (no extra wall time).

        3. DataStore profile with private members — the artifact target is a
           ds:// URL backed by a v3io profile carrying a private access key.
           The v1/v2 deploys both require the init container to resolve the
           profile (read its private body from env via the envFrom mount) and
           use the resolved key to authenticate to v3io.
        """
        access_key = os.environ.get("V3IO_ACCESS_KEY")
        assert access_key, "V3IO_ACCESS_KEY required for this system test"

        profile_name = "nuclio-store-test-v3io-profile"
        profile = datastore_profile.DatastoreProfileV3io(
            name=profile_name,
            v3io_access_key=access_key,
        )
        self.project.register_datastore_profile(profile)
        # Also register client-side so log_code_file can resolve ds:// URLs.
        datastore_profile.register_temporary_client_datastore_profile(profile)

        # Note: we write to .py files (not log_code_file(body=...)) so the
        # artifact's target_path keeps the .py extension — the loader does
        # importlib.import_module(<key>), which requires a matching .py file
        # in the source-loader output directory.
        artifact_key = "artifact_update_handler"
        artifact_target_path = (
            f"ds://{profile_name}/projects/{self.project_name}/code/{artifact_key}.py"
        )

        with tempfile.TemporaryDirectory(
            prefix="nuclio_store_artifact_update_"
        ) as temp_dir:
            handler_path = os.path.join(temp_dir, f"{artifact_key}.py")

            # Use an explicit tag so each re-log under the same key+tag updates
            # the tag's pointer to the new tree (instead of accumulating
            # untagged versions, which makes the store:// URI ambiguous and
            # breaks the server-side resolver with "Multiple rows found").
            tag = "current"

            with open(handler_path, "w") as f:
                f.write(self._versioned_handler_body("v1"))
            artifact_v1 = self.project.log_code_file(
                key=artifact_key,
                local_path=handler_path,
                tag=tag,
                target_path=artifact_target_path,
            )
            # Use the artifact's canonical URI rather than reconstructing it
            # — the stored format may evolve (e.g. tree-ref encoding) and
            # the test should follow whatever the system actually wrote.
            store_uri = artifact_v1.uri

            function = self.project.set_function(
                func=store_uri,
                name="nuclio-update-app",
                kind="nuclio",
                handler=f"{artifact_key}:handler",
                image=self._function_image,
            )

            self._logger.debug("First deploy with v1 source")
            function.deploy()
            assert function.status.state == "ready"
            self._assert_init_container_present(function)

            # The source-loader init container for store:// sources mounts
            # mlrun-project-secrets-<project> via envFrom, so the loader can
            # resolve DataStore profiles with private members and authenticate
            # to credential-protected datastores. Asserted on this deploy
            # (already being done) to avoid an extra wall-time test.
            init_containers = function.spec.config.get("spec.initContainers") or []
            loader = next(
                c
                for c in init_containers
                if c.get("name")
                == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
            )
            env_from = loader.get("envFrom") or []
            expected_secret = f"mlrun-project-secrets-{self.project_name}"
            assert any(
                e.get("secretRef", {}).get("name") == expected_secret for e in env_from
            ), (
                f"Init container missing envFrom secretRef to {expected_secret}; "
                f"got envFrom={env_from}"
            )

            v1_response = function.invoke("/")
            assert v1_response == {"version": "v1"}

            # Re-log under the same key+tag with v2 body — the tag pointer
            # moves to the new tree, store:// URI resolves to v2.
            with open(handler_path, "w") as f:
                f.write(self._versioned_handler_body("v2"))
            self.project.log_code_file(
                key=artifact_key,
                local_path=handler_path,
                tag=tag,
                target_path=artifact_target_path,
            )
            # The deploy clears spec.build.source server-side and stashes the
            # original URI in status.application_source so subsequent
            # (re)deploys can re-resolve it.
            assert function.status.application_source == store_uri

            self._logger.debug("Redeploying with v2 source")
            function.deploy()
            assert function.status.state == "ready"

            v2_response = function.invoke("/")
            assert v2_response == {"version": "v2"}

    def test_cross_runtime_same_artifact(self):
        """
        Same store:// CodeArtifact used by both job and nuclio runtimes:
        1. log_code_file → store://
        2. Job runtime: set_function(kind="job"), run_function() succeeds.
        3. Nuclio runtime: set_function(kind="nuclio"), deploy succeeds.

        Confirms the same artifact URI is consumable by both runtime paths
        and is preserved verbatim in spec.build.source for both.
        """
        # Use the job-style handler.py for the job side, echo handler for nuclio.
        # We cannot share one handler signature across job + nuclio (the call
        # contracts differ); the test asserts the *artifact URI* is consumable,
        # not that one Python file works in both runtimes.
        nuclio_uri = self._log_code_artifact("cross_runtime_nuclio_code")
        job_uri = self._log_code_artifact(
            "cross_runtime_job_code",
            src_path=os.path.join(self.assets_path, "handler.py"),
        )

        nuclio_fn = self.project.set_function(
            func=nuclio_uri,
            name="cross-runtime-nuclio",
            kind="nuclio",
            handler=self._function_handler,
            image=self._function_image,
        )
        assert nuclio_fn.spec.build.source == nuclio_uri

        job_fn = self.project.set_function(
            func=job_uri,
            name="cross-runtime-job",
            kind="job",
            handler="handler.my_func",
            image=self._function_image,
        )
        assert job_fn.spec.build.source == job_uri

        self._logger.debug("Deploying nuclio side of cross-runtime test")
        nuclio_fn.deploy()
        assert nuclio_fn.status.state == "ready"
        self._assert_init_container_present(nuclio_fn)

        self._logger.debug("Running job side of cross-runtime test")
        run = job_fn.run(params={"p1": 7})
        assert run.state() == "completed"
        assert run.outputs["accuracy"] == 14
