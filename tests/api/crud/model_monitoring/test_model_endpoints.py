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

from typing import Iterator
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session as DBSession

import mlrun.common.schemas
from mlrun.api.crud.model_monitoring.model_endpoints import ModelEndpoints
from mlrun.artifacts import ModelArtifact


@pytest.fixture
def db_session() -> DBSession:
    return Mock(spec=DBSession)


@pytest.fixture
def model_endpoint() -> mlrun.common.schemas.ModelEndpoint:
    return mlrun.common.schemas.ModelEndpoint(
        spec=mlrun.common.schemas.model_monitoring.ModelEndpointSpec(
            model_uri="some_fake_uri"
        )
    )


@pytest.fixture
def _patch_external_resources() -> Iterator[None]:
    with patch("mlrun.api.api.utils.get_run_db_instance", autospec=True):
        with patch(
            "mlrun.datastore.store_resources.get_store_resource",
            return_value=ModelArtifact(),
        ):
            with patch(
                "mlrun.api.crud.model_monitoring.model_endpoints.get_model_endpoint_store",
                autospec=True,
            ):
                yield


@pytest.mark.usefixtures("_patch_external_resources")
def test_create_with_empty_feature_stats(
    db_session: DBSession,
    model_endpoint: mlrun.common.schemas.ModelEndpoint,
) -> None:
    ModelEndpoints.create_model_endpoint(
        db_session=db_session, model_endpoint=model_endpoint
    )
