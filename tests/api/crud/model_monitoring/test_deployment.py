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
import typing
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import taosws

import mlrun.runtimes
import server
import server.api.crud.model_monitoring.deployment as mm_dep


@pytest.fixture()
def monitoring_deployment() -> mm_dep.MonitoringDeployment:
    return mm_dep.MonitoringDeployment(
        project=Mock(spec=str),
        auth_info=Mock(spec=mm_dep.mlrun.common.schemas.AuthInfo),
        db_session=Mock(spec=mm_dep.sqlalchemy.orm.Session),
        model_monitoring_access_key=None,
    )


class SecretTester(server.api.crud.secrets.Secrets):
    _secrets: dict[str, dict[str, str]] = {}

    def store_project_secrets(
        self,
        project: str,
        secrets: mlrun.common.schemas.SecretsData,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
        allow_storing_key_maps: bool = False,
    ):
        self._secrets[project] = secrets.secrets

    def get_project_secret(
        self,
        project: str,
        provider: mlrun.common.schemas.SecretProviderName,
        secret_key: str,
        token: typing.Optional[str] = None,
        allow_secrets_from_k8s: bool = False,
        allow_internal_secrets: bool = False,
        key_map_secret_key: typing.Optional[str] = None,
    ) -> typing.Optional[str]:
        return self._secrets.get(project, {}).get(secret_key)


class TestAppDeployment:
    """Test nominal flow of the app deployment"""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _patch_build_function() -> Iterator[None]:
        with patch(
            "server.api.utils.functions.build_function",
            new=Mock(return_value=(Mock(spec=mlrun.runtimes.ServingRuntime), True)),
        ):
            with patch("server.api.crud.Secrets", new=SecretTester):
                yield

    @staticmethod
    def test_app_dep(monitoring_deployment: mm_dep.MonitoringDeployment) -> None:
        monitoring_deployment.deploy_histogram_data_drift_app(
            image="mlrun/mlrun", overwrite=True
        )

    @staticmethod
    @pytest.fixture
    def store_connection(tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path / 'test.db'}"

    @pytest.mark.skipif(
        os.getenv("V3IO_FRAMESD") is None or os.getenv("V3IO_ACCESS_KEY") is None,
        reason="Configure Framsed to access V3IO store targets",
    )
    def test_credentials(
        self, monitoring_deployment: mm_dep.MonitoringDeployment, store_connection: str
    ) -> None:
        # new project case
        with patch(
            "server.api.crud.Functions.get_function",
            side_effect=mlrun.errors.MLRunNotFoundError,
        ):
            mlrun.mlconf.model_endpoint_monitoring.tsdb_connection = "v3io"
            with pytest.raises(mlrun.errors.MLRunBadRequestError):
                monitoring_deployment.check_if_credentials_are_set()

            with pytest.raises(mlrun.errors.MLRunInvalidMMStoreType):
                monitoring_deployment.set_credentials(
                    endpoint_store_connection="v3io",
                    stream_path="kafka://stream",
                    tsdb_connection="wrong",
                )

            with pytest.raises(taosws.QueryError):
                monitoring_deployment.set_credentials(
                    endpoint_store_connection="v3io",
                    stream_path="kafka://stream",
                    tsdb_connection="taosws://",
                )

            monitoring_deployment.set_credentials(
                endpoint_store_connection=store_connection,
                stream_path="v3io",
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
            assert (
                secrets[
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
                ]
                == store_connection
            )
            assert (
                secrets[
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION
                ]
                == "v3io"
            )

            monitoring_deployment.set_credentials(
                endpoint_store_connection=store_connection,
                tsdb_connection="v3io",
            )

            with pytest.raises(mlrun.errors.MLRunConflictError):
                monitoring_deployment.set_credentials(
                    endpoint_store_connection="v3io",
                    stream_path="v3io",
                    tsdb_connection="v3io",
                )
            monitoring_deployment.set_credentials(
                stream_path="kafka://stream",
                force=True,
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
            assert (
                secrets[
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
                ]
                == store_connection
            )
            assert (
                secrets[
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
                ]
                == "kafka://stream"
            )
        # existing project - upgrade from 1.6.0 case
        monitoring_deployment.project = "1.6.0_project"
        with patch(
            "server.api.crud.Functions.get_function", return_value=Mock(spec={})
        ):
            with pytest.raises(mlrun.errors.MLRunConflictError):
                monitoring_deployment.set_credentials(
                    endpoint_store_connection=store_connection,
                    stream_path="v3io",
                    tsdb_connection="v3io",
                )
            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
            for key, value in secrets.items():
                if (
                    key
                    != mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
                ):
                    assert value == "v3io"
            monitoring_deployment.set_credentials(
                endpoint_store_connection=store_connection,
                stream_path="v3io",
                tsdb_connection="v3io",
                force=True,
            )

            secrets = monitoring_deployment._get_monitoring_mandatory_project_secrets()
            assert (
                secrets[
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
                ]
                == store_connection
            )
