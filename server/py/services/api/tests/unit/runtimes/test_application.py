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

import typing

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.nuclio.function
import mlrun.runtimes.pod

import services.api.crud.runtimes.nuclio.function
import services.api.crud.runtimes.nuclio.helpers
from services.api.tests.unit.runtimes.base import TestRuntimeBase


class TestApplicationRuntime(TestRuntimeBase):
    @property
    def runtime_kind(self):
        # enables extending classes to run the same tests with different runtime
        return "application"

    @property
    def class_name(self):
        # enables extending classes to run the same tests with different class
        return "application"

    def test_compile_function_config_skipped_spec(
        self, db: Session, client: TestClient
    ):
        """
        Test that compiling function configuration with requirements and base image are skipped
        """
        function = self._generate_runtime(self.runtime_kind)
        requirements = ["requests", "numpy"]
        function.with_requirements(requirements=requirements)
        function.spec.build.base_image = "my-base-image"
        function.spec.build.source = "v3io://my-source.tar.gz"
        (
            _,
            _,
            config,
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function, builder_env={}
        )
        assert not mlrun.utils.get_in(
            config,
            "spec.build.commands",
        )
        assert not mlrun.utils.get_in(
            config,
            "spec.build.baseImage",
        )
        assert not mlrun.utils.get_in(
            config,
            "spec.build.codeEntryType",
        )

    def test_create_function_validate_min_nuclio_version(
        self, db: Session, client: TestClient
    ):
        """Verify that the nuclio min version is validated by the ApplicationRuntime constructor"""
        mlrun.mlconf.nuclio_version = "1.12.14"
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError) as exc:
            self._generate_runtime(self.runtime_kind)
        assert (
            str(exc.value)
            == "'Application Runtime' function requires Nuclio v1.13.1 or higher"
        )

    def test_http_trigger_configuration(self):
        runtime = self._generate_runtime(self.runtime_kind)
        assert (
            runtime.spec.config.get("spec.triggers.application-http", {}).get(
                "maxWorkers"
            )
            == mlrun.mlconf.function.application.default_worker_number
        )
        # replace an http trigger name to simulate custom http trigger
        runtime.spec.config["spec.triggers.application-http-copy"] = (
            runtime.spec.config.get("spec.triggers.application-http")
        )
        runtime.spec.config.pop("spec.triggers.application-http")
        runtime._ensure_reverse_proxy_configurations()
        # ensure default application-http is not added as part of enrichment
        assert runtime.spec.config.get("spec.triggers.application-http") is None

    @pytest.mark.parametrize(
        "source,load_source_on_run,expected",
        [
            # Store URIs always need init container
            ("store://artifacts/project/my-source", False, True),
            ("store://artifacts/project/my-source", True, True),
            # Git/archive need init container only with pull_at_runtime=True
            ("git://github.com/org/repo.git", True, True),
            ("git://github.com/org/repo.git", False, False),
            ("https://example.com/source.tar.gz", True, True),
            ("https://example.com/source.tar.gz", False, False),
            ("https://example.com/source.zip", True, True),
            ("https://example.com/source.zip", False, False),
            # Non-matching sources don't need init container
            ("https://example.com/file.py", False, False),
            ("https://example.com/file.py", True, False),
            # No source
            (None, False, False),
            ("", False, False),
        ],
    )
    def test_should_fetch_source_code(self, source, load_source_on_run, expected):
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.source = source
        function.spec.build.load_source_on_run = load_source_on_run

        result = services.api.crud.runtimes.nuclio.function._should_fetch_source_code(
            function
        )
        assert result == expected

    def test_configure_init_container_with_store_uri(
        self, db: Session, client: TestClient
    ):
        # Test init container is configured correctly for store URI source
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.source = "store://artifacts/test-project/my-source"

        services.api.crud.runtimes.nuclio.function._compile_function_config(
            function, builder_env={}
        )

        # Verify init container exists
        init_containers = function.spec.config.get("spec.initContainers", [])
        assert len(init_containers) == 1
        assert (
            init_containers[0]["name"]
            == mlrun.common.constants.SOURCE_LOADER_INIT_CONTAINER_NAME
        )
        assert init_containers[0]["command"] == ["mlrun", "load-source"]
        assert init_containers[0]["image"]
        assert "store://artifacts/test-project/my-source" in init_containers[0]["args"]

        # Assert init container mounts the volume to the target dir
        init_mounts = init_containers[0].get("volumeMounts", [])
        assert len(init_mounts) == 1
        assert init_mounts[0]["name"] == mlrun.common.constants.SOURCE_CODE_VOLUME_NAME
        assert (
            init_mounts[0]["mountPath"]
            == mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
        )

        # Verify sidecar configuration
        sidecars = function.spec.config.get("spec.sidecars", [])
        assert len(sidecars) == 1
        assert (
            sidecars[0].get("workingDir")
            == mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
        )
        assert any(e.get("name") == "PYTHONPATH" for e in sidecars[0].get("env", []))

        # Sidecar has volume mount
        sidecar_mounts = sidecars[0].get("volumeMounts", [])
        assert any(
            vm.get("name") == mlrun.common.constants.SOURCE_CODE_VOLUME_NAME
            and vm.get("mountPath")
            == mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
            for vm in sidecar_mounts
        )

    def test_configure_init_container_custom_target_dir(
        self, db: Session, client: TestClient
    ):
        # Test init container respects custom source_code_target_dir
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.source = "store://artifacts/test-project/my-source"
        function.spec.build.source_code_target_dir = "/custom/path"

        services.api.crud.runtimes.nuclio.function._compile_function_config(
            function, builder_env={}
        )

        init_containers = function.spec.config.get("spec.initContainers", [])
        assert "/custom/path" in init_containers[0]["args"]

        # Assert that the volume mount uses the custom path
        assert init_containers[0]["volumeMounts"][0]["mountPath"] == "/custom/path"

        sidecars = function.spec.config.get("spec.sidecars", [])
        assert sidecars[0].get("workingDir") == "/custom/path"

    def test_configure_init_container_idempotent(self, db: Session, client: TestClient):
        """
        Verify that repeated compilation does not duplicate init containers, volume mounts, or environment variables.
        """
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.source = "store://artifacts/test-project/my-source"

        # Run compile multiple times to simulate repeated deploys
        for _ in range(3):
            services.api.crud.runtimes.nuclio.function._compile_function_config(
                function, builder_env={}
            )

        # Verify only one init container configured
        init_containers = function.spec.config.get("spec.initContainers", [])
        assert len(init_containers) == 1

        # Volume mount is not duplicated inside the init container
        init_mounts = init_containers[0].get("volumeMounts", [])
        assert len(init_mounts) == 1
        assert init_mounts[0]["name"] == mlrun.common.constants.SOURCE_CODE_VOLUME_NAME

        # Verify sidecar has only one PYTHONPATH and one source volume mount
        sidecars = function.spec.config.get("spec.sidecars", [])
        assert len(sidecars) == 1

        # PYTHONPATH is added once
        sidecar_env = sidecars[0].get("env", [])
        pythonpath_entries = [e for e in sidecar_env if e.get("name") == "PYTHONPATH"]
        assert len(pythonpath_entries) == 1
        assert (
            pythonpath_entries[0]["value"]
            == mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
        )

        # Volume mount is added once to sidecar
        sidecar_mounts = sidecars[0].get("volumeMounts", [])
        source_mounts = [
            vm
            for vm in sidecar_mounts
            if vm.get("name") == mlrun.common.constants.SOURCE_CODE_VOLUME_NAME
        ]
        assert len(source_mounts) == 1
        assert (
            source_mounts[0]["mountPath"]
            == mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR
        )

        # Verify PYTHONPATH prepends to existing value instead of skipping
        # Reset function and pre-set a custom PYTHONPATH
        function2 = self._generate_runtime(self.runtime_kind)
        function2.spec.build.source = "store://artifacts/test-project/my-source"
        sidecars2 = function2.spec.config.get("spec.sidecars", [])
        sidecars2[0].setdefault("env", []).append(
            {"name": "PYTHONPATH", "value": "/user/custom/path"}
        )

        services.api.crud.runtimes.nuclio.function._compile_function_config(
            function2, builder_env={}
        )

        sidecar_env2 = sidecars2[0].get("env", [])
        pythonpath2 = next(e for e in sidecar_env2 if e.get("name") == "PYTHONPATH")
        assert pythonpath2["value"] == (
            f"{mlrun.common.constants.DEFAULT_SOURCE_CODE_TARGET_DIR}:/user/custom/path"
        )

    @pytest.mark.parametrize(
        "source,load_source_on_run",
        [
            # No source
            (None, False),
            ("", False),
            # Not a store:// URI
            ("https://example.com/file.py", False),
            ("https://example.com/file.py", True),
            # Git without pull-at-runtime
            ("git://github.com/org/repo.git", False),
            # Archive without pull-at-runtime
            ("https://example.com/source.tar.gz", False),
            ("https://example.com/source.zip", False),
            # Local path
            ("/local/path/app.py", False),
        ],
    )
    def test_no_init_container_for_non_matching_sources(
        self, db: Session, client: TestClient, source, load_source_on_run
    ):
        # Verify that init container is not configured when source does not require runtime loading.
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.source = source
        function.spec.build.load_source_on_run = load_source_on_run

        services.api.crud.runtimes.nuclio.function._compile_function_config(
            function, builder_env={}
        )

        # Init container must not exist
        init_containers = function.spec.config.get("spec.initContainers", [])
        assert len(init_containers) == 0

        # Sidecar must not be patched
        sidecars = function.spec.config.get("spec.sidecars", [])

        # workingDir should not be forced to source target dir
        assert sidecars[0].get("workingDir") is None

        # PYTHONPATH should not be injected
        sidecar_env = sidecars[0].get("env", [])
        assert not any(e.get("name") == "PYTHONPATH" for e in sidecar_env)

        # Source volume mount should not exist
        sidecar_mounts = sidecars[0].get("volumeMounts", [])
        assert not any(
            vm.get("name") == mlrun.common.constants.SOURCE_CODE_VOLUME_NAME
            for vm in sidecar_mounts
        )

    def _execute_run(self, runtime, **kwargs):
        # deploy_nuclio_function doesn't accept watch, so we need to remove it
        kwargs.pop("watch", None)
        services.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
            runtime, **kwargs
        )

    def _generate_runtime(
        self, kind=None
    ) -> typing.Union[mlrun.runtimes.ApplicationRuntime]:
        runtime = mlrun.new_function(
            name=self.name,
            project=self.project,
            kind=kind or self.runtime_kind,
        )
        runtime._ensure_reverse_proxy_configurations()
        runtime._configure_application_sidecar()
        return runtime
