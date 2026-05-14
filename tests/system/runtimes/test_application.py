# Copyright 2024 Iguazio
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

import io
import os
import shutil
import sys
import tempfile

import pytest
from nuclio.auth import AuthInfo as NuclioAuthInfo

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.datastore.datastore_profile as datastore_profile
import mlrun.runtimes
import mlrun.runtimes.utils
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestApplicationRuntime(tests.system.base.TestMLRunSystem):
    project_name = "application-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._vizro_app_code_filename = "vizro_app.py"
        self._function_with_delay_healthcheck = "function_with_delay_healthcheck.py"
        self._simple_flask_app = "simple_flask_app.py"
        self._source_archive = "source_archive.tar.gz"
        self._files_to_upload = [
            self._vizro_app_code_filename,
            self._function_with_delay_healthcheck,
            self._simple_flask_app,
            self._source_archive,
        ]
        self._source = os.path.join(self.remote_code_dir, self._vizro_app_code_filename)

    def test_deploy_application(self):
        self._upload_code_to_cluster()

        self._logger.debug("Creating application")
        function, source = self._create_vizro_application()

        self._logger.debug("Deploying vizro application")
        function.deploy(with_mlrun=False)

        assert function.invoke("/", verify=False)

        # Application runtime function is created without external url
        # check that empty string is not added to func.status.external_invocation_urls
        assert "" not in function.status.external_invocation_urls

        assert function.spec.build.source == source
        assert (
            function.status.application_image
            == f".mlrun/func-{self.project.metadata.name}-{function.metadata.name}:latest"
        )

        # Assert get state does not create a new function since the state hasn't changed
        db_functions = self._run_db.list_functions(name="vizro-app")
        current_functions_in_db = len(db_functions)

        # Run get state multiple times to make sure it doesn't create a new function
        for i in range(5):
            function._get_state()
        db_functions = self._run_db.list_functions(name="vizro-app")
        assert len(db_functions) == current_functions_in_db

        self._logger.debug("Redeploying the same application with capturing stdout")
        output = self._deploy_application_with_stdout_capture(function)

        # Assert nuclio image build was skipped
        assert "(info) Skipping build" in output

        assert function.invoke("/", verify=False)
        assert function.spec.build.source == source
        assert (
            function.status.application_image
            == f".mlrun/func-{self.project.metadata.name}-{function.metadata.name}:latest"
        )

    def test_deploy_application_from_image(self):
        self._logger.debug("Creating first application")
        function, source = self._create_vizro_application(name="first-app")

        self._logger.debug("Deploying first application")
        function.deploy(with_mlrun=False)

        assert function.invoke("/", verify=False)

        # take the application image and container image, and use them to deploy a new function
        application_image = function.status.application_image
        container_image = function.status.container_image

        function, _ = self._create_vizro_application(
            name="second-app", app_image=application_image
        )
        function.from_image(container_image)

        self._logger.debug("Deploying a second application")
        output = self._deploy_application_with_stdout_capture(function)

        # make sure the build was skipped
        assert "(info) Skipping build" in output

    def test_deploy_application_from_project_source(self):
        self._upload_code_to_cluster()

        self.project.set_source(self._source)
        self.project.save()

        self._logger.debug("Creating application")
        function, source = self._create_vizro_application(with_repo=True)

        self._logger.debug("Deploying vizro application")
        function.deploy(with_mlrun=False)

        assert function.invoke("/", verify=False)

    def test_deploy_reverse_proxy_base_image(self):
        tests.system.base.TestMLRunSystem._logger.debug(
            "Deploying reverse proxy base image"
        )
        mlrun.runtimes.ApplicationRuntime.deploy_reverse_proxy_image()
        assert mlrun.runtimes.ApplicationRuntime.reverse_proxy_image

        # deploy an application and expect it to use the reverse proxy image
        function, source = self._create_vizro_application()

        self._logger.debug("Deploying vizro application")
        function.deploy(with_mlrun=False)

        assert (
            function.status.container_image
            == mlrun.runtimes.ApplicationRuntime.reverse_proxy_image
        )

        assert (
            function.metadata.annotations.get("kubectl.kubernetes.io/default-container")
            == function.status.sidecar_name
        )

    @pytest.mark.enterprise
    def test_deploy_application_with_custom_api_gateway(self):
        self._upload_code_to_cluster()

        self._logger.debug("Creating application")
        function, source = self._create_vizro_application()

        self._logger.debug("Deploying vizro application")
        function.deploy(with_mlrun=False, create_default_api_gateway=False)

        auth = NuclioAuthInfo(username="my-user", password="123").to_requests_auth()
        function.create_api_gateway(
            name="my-api-gateway",
            authentication_mode=mlrun.common.schemas.APIGatewayAuthenticationMode.basic,
            authentication_creds=(auth.username, auth.password),
        )

        assert function.invoke("/", verify=False, auth=auth)
        with pytest.raises(RuntimeError, match="401 Authorization Required"):
            function.invoke("/", verify=False)

        # Change API gateway to access key mode
        function.create_api_gateway(
            name="my-api-gateway",
            authentication_mode=mlrun.common.schemas.APIGatewayAuthenticationMode.access_key,
        )

        # Invoke with access key
        auth = NuclioAuthInfo().from_envvar().to_requests_auth()
        assert function.invoke("/", verify=False, auth=auth)
        with pytest.raises(RuntimeError, match="401 Authorization Required"):
            function.invoke("/", verify=False)

        # Create API gateway with new name and set it as default
        api_gateway_name = "my-other-api-gateway"
        function.create_api_gateway(
            name=api_gateway_name,
            authentication_mode=mlrun.common.schemas.APIGatewayAuthenticationMode.access_key,
            set_as_default=True,
        )
        # Invoke should infer access key is needed
        assert function.invoke("/", verify=False)
        assert function.status.api_gateway_name == api_gateway_name
        # At this point we are yet to get the function status from the server so the external invocation URL contains
        # only "my-api-gateway"
        assert len(function.status.external_invocation_urls) == 1

        function._get_state()
        assert len(function.status.external_invocation_urls) == 2

    def test_application_probes(self):
        self._upload_code_to_cluster()
        self._logger.debug("Creating application")
        function = self._create_delay_health_check_application("delay-health-check-app")
        # Add probes to the function - set timeout_seconds > 20 to allow health endpoint to respond
        self._logger.debug("Adding probes to function")
        function.set_probe(
            type="readiness",
            http_path="/health",
            http_port=5000,  # Flask app runs on port 5000
            period_seconds=5,
            timeout_seconds=25,  # Wait 25 seconds (more than the 20 seconds sleep in /health)
        )

        # Deploy with probes
        self._logger.debug("Deploying application with probes")
        function.deploy(with_mlrun=False)

        # Verify the application is running and healthy
        self._logger.debug("Validating application is healthy")
        response = function.invoke("/external", verify=False).content.decode("utf-8")
        assert response == "test message"

    def test_deploy_application_with_source_reload(self):
        """
        Test that application source code can be updated and reloaded at runtime using the init container.
        """
        # Work on a temporary copy of the source to avoid mutating shared test assets
        original_path = os.path.join(self.assets_path, self._simple_flask_app)

        with tempfile.TemporaryDirectory(prefix="source_reload_test_") as temp_dir:
            source_path = os.path.join(temp_dir, self._simple_flask_app)
            shutil.copy(original_path, source_path)

            function = self._create_simple_flask_application(
                name="source-reload-app", source=source_path
            )
            function.set_probe(type="readiness", http_path="/health", period_seconds=2)

            # First deploy - auto-uploads source as artifact
            self._logger.debug("Deploying application with version-1 source")
            function.deploy(with_mlrun=False)

            # Invoke and verify version-1
            response = function.invoke("/", verify=False)
            assert response.content.decode("utf-8") == "version-1"

            # Update source to version-2 by modifying the VERSION constant
            self._logger.debug("Updating source to version-2")
            with open(source_path) as f:
                source_v2 = f.read().replace(
                    'VERSION = "version-1"', 'VERSION = "version-2"'
                )
            with open(source_path, "w") as f:
                f.write(source_v2)

            # Redeploy - auto-uploads new source, init container loads it without image rebuild
            build_pod_before = function.status.build_pod
            self._logger.debug("Redeploying with version-2 source")
            function.deploy(with_mlrun=False)

            # Verify sidecar image was not rebuilt (only source changed, no new build pod)
            assert function.status.build_pod == build_pod_before

            # Invoke and verify version-2
            response = function.invoke("/", verify=False)
            assert response.content.decode("utf-8") == "version-2"

    def test_deploy_application_with_git_source(self):
        """
        Test that application runtime uses init container to load Git source at runtime.
        Verifies that:
        1. Source is cloned from Git without being built into the image
        2. Git repo files are accessible (both root and subdir)
        3. Sidecar image is not rebuilt on redeployment
        4. Workdir changes the sidecar's working directory to a subdirectory
        """
        git_url = "git://github.com/mlrun/test-git-load.git#main"

        self._logger.debug("Creating application with Git source")
        function = self.project.set_function(
            name="git-app",
            kind="application",
            image="python:3.11",
        )
        function.with_source_archive(
            source=git_url,
            pull_at_runtime=True,
        )
        function.spec.command = "python"
        function.spec.args = ["-m", "http.server", "8050"]
        function.set_internal_application_port(8050)
        function.set_probe(type="readiness", http_path="/", period_seconds=2)

        # First deploy
        self._logger.debug("First deploy with Git source and pull_at_runtime=True")
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"

        # Verify Git repo was cloned and files are accessible
        response = function.invoke("/subdir/mylib.py", verify=False)
        assert response.status_code == 200

        # Verify a root level file is also accessible
        response = function.invoke("/rootlib.py", verify=False)
        assert response.status_code == 200

        # Redeploy - sidecar image should not be rebuilt
        build_pod_before = function.status.build_pod
        self._logger.debug("Redeploying - sidecar build should be skipped")
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"
        assert function.status.build_pod == build_pod_before

        # Verify source is still accessible after redeploy
        response = function.invoke("/subdir/mylib.py", verify=False)
        assert response.status_code == 200

        # force_build=True should trigger sidecar image build (new build pod created)
        self._logger.debug("Redeploying with force_build=True")
        function.deploy(with_mlrun=False, force_build=True)
        assert function.status.state == "ready"
        assert function.status.build_pod != build_pod_before

        # Verify source is still accessible after forced rebuild
        response = function.invoke("/subdir/mylib.py", verify=False)
        assert response.status_code == 200

        # Redeploy with workdir='subdir' - sidecar should now serve from
        # /home/mlrun_code/subdir/ instead of /home/mlrun_code/
        self._logger.debug("Redeploying with workdir='subdir'")
        function.with_source_archive(
            source=git_url,
            workdir="subdir",
            pull_at_runtime=True,
        )
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"

        # mylib.py is now at root path (served from subdir/)
        response = function.invoke("/mylib.py", verify=False)
        assert response.status_code == 200

        # rootlib.py is no longer accessible (working dir is subdir, not repo root)
        response = function.invoke("/rootlib.py", verify=False)
        assert response.status_code == 404

    def test_deploy_application_with_archive_source(self):
        """
        Test that application runtime uses init container to load archive source at runtime.
        Verifies that:
        1. Archive is extracted without being built into the image
        2. Extracted files are accessible
        3. Sidecar image is not rebuilt on redeployment
        """
        # Upload archive to remote storage so init container can access it
        self._upload_code_to_cluster()
        archive_url = os.path.join(self.remote_code_dir, self._source_archive)

        self._logger.debug("Creating application with archive source")
        function = self.project.set_function(
            name="archive-app",
            kind="application",
            image="python:3.11",
        )
        function.with_source_archive(
            source=archive_url,
            pull_at_runtime=True,
        )
        function.spec.command = "python"
        function.spec.args = ["-m", "http.server", "8050"]
        function.set_internal_application_port(8050)
        function.set_probe(type="readiness", http_path="/", period_seconds=2)

        # First deploy
        self._logger.debug("First deploy with archive source and pull_at_runtime=True")
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"

        # Verify extracted archive files are accessible
        response = function.invoke("/rootlib.py", verify=False)
        assert response.status_code == 200

        # Redeploy - sidecar image should not be rebuilt
        build_pod_before = function.status.build_pod
        self._logger.debug("Redeploying - sidecar build should be skipped")
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"
        assert function.status.build_pod == build_pod_before

    def _create_vizro_application(
        self, name="vizro-app", app_image=None, with_repo: bool = False
    ):
        function = self.project.set_function(
            name=name,
            kind="application",
            requirements=["vizro<0.1.32", "gunicorn", "Werkzeug==2.2.2"],
            with_repo=with_repo,
        )
        function.set_internal_application_port(8050)
        function.spec.command = "gunicorn"
        function.spec.args = [
            "vizro_app:server",
            "--bind",
            "0.0.0.0:8050",
            "--log-level",
            "debug",
        ]
        if app_image:
            function.spec.image = app_image
        elif not with_repo:
            function.with_source_archive(source=self._source)
        return function, self._source

    @staticmethod
    def _deploy_application_with_stdout_capture(function):
        # Create a StringIO object to capture stdout
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            function.deploy(with_mlrun=False)
        finally:
            sys.stdout = old_stdout
        output = new_stdout.getvalue()
        new_stdout.close()
        return output

    def _create_delay_health_check_application(self, name="delay-app"):
        function = self.project.set_function(
            name=name,
            kind="application",
            requirements=["Flask==3.0.0"],
            with_repo=False,
        )
        function.set_internal_application_port(5000)  # Match Flask port
        function.spec.command = "python"
        function.spec.args = [
            "-m",
            "flask",
            "--app=function_with_delay_healthcheck",
            "run",
            "--host=0.0.0.0",
            "--port=5000",
        ]
        function.with_http(workers=1, trigger_name="application-http")
        delay_healthcheck_source = os.path.join(
            self.remote_code_dir, self._function_with_delay_healthcheck
        )
        function.with_source_archive(source=delay_healthcheck_source)
        return function

    def test_deploy_application_with_store_uri_via_datastore_profile(self):
        """End-to-end: an Application function whose store:// CodeArtifact
        target_path resolves through a DataStore profile with private members
        deploys successfully.

        Verifies that the source-loader init container can resolve the profile
        at fetch time — which requires the profile's private body to be readable
        from env via the project-secret envFrom mount on the init container.

        Without the envFrom mount on Application's init container, the deploy
        would fail with MLRunNotFoundError during artifact resolution (the
        profile's private body would be absent from the init container's env).
        """
        access_key = os.environ.get("V3IO_ACCESS_KEY")
        assert access_key, "V3IO_ACCESS_KEY required for this system test"

        profile_name = "application-store-test-v3io-profile"
        profile = datastore_profile.DatastoreProfileV3io(
            name=profile_name,
            v3io_access_key=access_key,
        )
        self.project.register_datastore_profile(profile)
        # Also register client-side so log_code_file can resolve the ds:// URL.
        datastore_profile.register_temporary_client_datastore_profile(profile)

        # Log the simple flask app as a CodeArtifact whose target_path lives
        # on the ds:// profile — the init container will resolve the profile
        # at fetch time, which requires the project-secret envFrom mount.
        artifact_key = "simple_flask_app_via_profile"
        local_path = os.path.join(self.assets_path, self._simple_flask_app)
        artifact_target_path = (
            f"ds://{profile_name}"
            f"/projects/{self.project_name}/code/{self._simple_flask_app}"
        )
        artifact = self.project.log_code_file(
            key=artifact_key,
            local_path=local_path,
            target_path=artifact_target_path,
        )
        # Use the artifact's canonical URI — see TESTING_STANDARDS.md §7.
        store_uri = artifact.uri

        function = self.project.set_function(
            func=store_uri,
            name="store-app-via-profile",
            kind="application",
            requirements=["Flask==3.0.0"],
        )
        function.set_internal_application_port(5000)
        function.spec.command = "python"
        function.spec.args = [
            "-m",
            "flask",
            "--app=simple_flask_app",
            "run",
            "--host=0.0.0.0",
            "--port=5000",
        ]

        self._logger.debug(
            "Deploying Application with store:// + DataStore profile source"
        )
        function.deploy(with_mlrun=False)
        assert function.status.state == "ready"

        # The source-loader init container must have the project-secret envFrom
        # mount, otherwise the deploy would have failed at profile resolution.
        # Asserting the spec wiring catches future regressions of the gating.
        init_containers = function.spec.config.get("spec.initContainers") or []
        loader = next(
            c
            for c in init_containers
            if c.get("name") == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
        )
        env_from = loader.get("envFrom") or []
        expected_secret = f"mlrun-project-secrets-{self.project_name}"
        assert any(
            e.get("secretRef", {}).get("name") == expected_secret for e in env_from
        ), (
            f"Init container missing envFrom secretRef to {expected_secret}; "
            f"got envFrom={env_from}"
        )

        # Functional check — the deployed flask app responds.
        response = function.invoke("/", verify=False)
        assert response.content.decode("utf-8") == "version-1"

    def _create_simple_flask_application(self, name="simple-flask-app", source=None):
        """Create a simple Flask application for testing source reload."""
        source_path = source or os.path.join(self.assets_path, self._simple_flask_app)
        function = self.project.set_function(
            func=source_path,
            name=name,
            kind="application",
            requirements=["Flask==3.0.0"],
        )
        function.set_internal_application_port(5000)
        function.spec.command = "python"
        function.spec.args = [
            "-m",
            "flask",
            "--app=simple_flask_app",
            "run",
            "--host=0.0.0.0",
            "--port=5000",
        ]
        return function
