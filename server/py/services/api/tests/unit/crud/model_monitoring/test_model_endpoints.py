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


import pytest
import sqlalchemy

import mlrun.common.schemas

from services.api.crud.model_monitoring.model_endpoints import ModelEndpoints


@pytest.fixture
def model_endpoint() -> mlrun.common.schemas.ModelEndpoint:
    return mlrun.common.schemas.ModelEndpoint(
        metadata=mlrun.common.schemas.model_monitoring.ModelEndpointMetadata(
            project="my-proj",
            name="my-endpoint",
        ),
        spec=mlrun.common.schemas.model_monitoring.ModelEndpointSpec(
            function_name="my-func", function_tag="my-tag"
        ),
        status=mlrun.common.schemas.model_monitoring.ModelEndpointStatus(),
    )


def test_create_with_empty_feature_stats(
    db: sqlalchemy.orm.Session,
    model_endpoint: mlrun.common.schemas.ModelEndpoint,
) -> None:
    ModelEndpoints().create_model_endpoint(
        db_session=db, model_endpoint=model_endpoint, creation_strategy="inplace"
    )
