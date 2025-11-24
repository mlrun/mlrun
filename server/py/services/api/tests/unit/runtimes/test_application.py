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
