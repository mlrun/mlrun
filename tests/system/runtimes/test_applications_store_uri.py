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

import pytest

import mlrun.common.constants
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestApplicationStoreUri(tests.system.base.TestMLRunSystem):
    """End-to-end coverage for kind=application functions whose source is an
    explicit store:// CodeArtifact URI.

    The application runtime supports store:// via a server-installed init
    container ('mlrun load-source') that downloads code into a shared volume
    before the user's sidecar container starts. This is pre-existing
    infrastructure - already in development - that pr-b shipped without an
    end-to-end test using an explicit store:// URI. (The auto-upload path is
    covered by test_deploy_application_with_source_reload; this test covers
    the explicit-URI path users hit when integrating with external code
    artifact systems like Vera.)
    """

    project_name = "application-store-uri-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._app_filename = "simple_flask_app.py"
        self._function_image = os.environ.get("MLRUN_TEST_IMAGE", "mlrun/mlrun")

    def _log_code_artifact(self, key: str) -> str:
        """Log the assets/simple_flask_app.py file as a CodeArtifact and
        return its canonical store:// URI.

        The asset (``tests/system/runtimes/assets/simple_flask_app.py``)
        is the existing Flask app used by other system tests in this
        directory — it returns ``"version-1"`` from ``/`` which the
        invocation assertion below pins.

        Uses the artifact's own ``.uri`` rather than rebuilding
        ``f"store://artifacts/<project>/<key>"`` from a template — the
        reconstructed value can drift from what the system actually
        stored. See TESTING_STANDARDS §7.
        """
        local_path = os.path.join(self.assets_path, self._app_filename)
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

    def test_e2e_application_function_from_store_artifact(self):
        """Application kind + explicit store:// URI.

        Flow:
          1. log_code_file uploads simple_flask_app.py as a CodeArtifact.
          2. set_function with kind="application" and func=<store-uri>.
          3. deploy_function - server installs the source-loader init
             container; the init container downloads the code at pod start.
          4. Asserts the init container is wired and the application
             responds.
        """
        store_uri = self._log_code_artifact("app_code")

        function = self.project.set_function(
            func=store_uri,
            name="app-from-store",
            kind="application",
            requirements=["Flask==3.0.0"],
            image=self._function_image,
        )
        function.set_internal_application_port(5000)
        function.spec.command = "python"
        function.spec.args = [
            "-m",
            "flask",
            f"--app={os.path.splitext(self._app_filename)[0]}",
            "run",
            "--host=0.0.0.0",
            "--port=5000",
        ]
        function.set_probe(type="readiness", http_path="/", period_seconds=2)

        function.deploy(with_mlrun=False)

        # store:// must remain in the function spec after deploy.
        db_function = self.project.get_function("app-from-store")
        assert db_function.spec.build.source == store_uri, (
            f"Expected store:// URI preserved in DB, got "
            f"{db_function.spec.build.source!r}"
        )

        self._assert_init_container_present(db_function)

        # Functional check: the app is reachable AND the init container
        # actually downloaded the right code. simple_flask_app.py returns
        # "version-1" from /; if the init container failed to download, the
        # sidecar wouldn't have the module and Flask would 500.
        response = function.invoke("/", verify=False)
        assert response.content.decode("utf-8") == "version-1"

    def test_e2e_application_missing_store_artifact_marks_error(self):
        """ML-12562: a deploy whose store:// source can't be resolved must not leave
        the function stuck "ready". The build phase stamps a premature "ready"; the
        deploy then fails at enrich (before Nuclio), and the status is reconciled to
        "error" since no Nuclio function is running.
        """
        bad_uri = f"store://artifacts/{self.project_name}/does-not-exist"

        function = self.project.set_function(
            func=bad_uri,
            name="app-missing-artifact",
            kind="application",
            # the requirement forces the image build that stamps the premature "ready"
            requirements=["Flask==3.0.0"],
            image=self._function_image,
        )
        function.set_internal_application_port(5000)
        function.spec.command = "python"
        function.spec.args = [
            "-m",
            "flask",
            "--app=does_not_exist",
            "run",
            "--host=0.0.0.0",
            "--port=5000",
        ]
        function.set_probe(type="readiness", http_path="/", period_seconds=2)

        with pytest.raises(Exception):
            function.deploy(with_mlrun=False)

        # ignore_cache=True forces a DB read - the in-memory object still holds the
        # build phase's premature "ready".
        db_function = self.project.get_function(
            "app-missing-artifact", ignore_cache=True
        )
        assert db_function.status.state == "error", (
            f"expected 'error', got {db_function.status.state!r}"
        )
