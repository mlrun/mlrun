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
import os
import pathlib

import fastapi
import pytest

from mlrun import mlconf

import framework.service
import services.alerts.daemon
from framework.tests.unit.common_fixtures import TestServiceBase
from services.alerts.daemon import daemon

tests_root_directory = pathlib.Path(__file__).absolute().parent
assets_path = tests_root_directory.joinpath("assets")

if str(tests_root_directory) in os.getcwd():
    # If this is the top level conftest - we need to explicitly declare the base common fixtures to
    # make pytest use them. If this is not the top level conftest (e.g. when running the tests from the project root)
    # then providing pytest_plugins is not allowed.
    pytest_plugins = [
        "tests.common_fixtures",
    ]


class TestAlertsBase(TestServiceBase):
    @pytest.fixture(scope="module")
    def app(self) -> fastapi.FastAPI:
        mlconf.services.service_name = "alert"
        mlconf.services.hydra.services = ""
        yield services.alerts.daemon.app()

    @pytest.fixture(scope="module")
    def prefix(self):
        yield daemon.service.base_versioned_service_prefix

    @pytest.fixture(autouse=True)
    def alert_config_test(self, service_config_test):
        # we need to override the container manually because we run all unit tests in
        # the same process in CI so services are imported even when they are not needed
        service_container = framework.service.ServiceContainer()
        service_container.override(services.alerts.daemon.AlertsServiceContainer)
        yield
        service_container.reset_override()
