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

from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest

import mlrun.runtimes
import server.api.crud.model_monitoring.deployment as mm_dep


@pytest.fixture()
def monitoring_deployment() -> mm_dep.MonitoringDeployment:
    return mm_dep.MonitoringDeployment(
        project=Mock(spec=str),
        auth_info=Mock(spec=mm_dep.mlrun.common.schemas.AuthInfo),
        db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
        model_monitoring_access_key=None,
    )


class TestAppDeployment:
    """Test nominal flow of the app deployment"""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _patch_build_function() -> Iterator[None]:
        with patch(
            "server.api.utils.functions.build_function",
            new=Mock(return_value=(Mock(spec=mlrun.runtimes.ServingRuntime), True)),
        ):
            yield

    @staticmethod
    def test_app_dep(monitoring_deployment: mm_dep.MonitoringDeployment) -> None:
        monitoring_deployment.deploy_histogram_data_drift_app(image="mlrun/mlrun")
