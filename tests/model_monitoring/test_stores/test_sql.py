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

import datetime
import string
import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from random import choice, randint
from typing import Optional, Union, cast
from zoneinfo import ZoneInfo

import pytest

import mlrun.common.schemas
import mlrun.model_monitoring
from mlrun.model_monitoring.db.stores.sqldb.sql_store import SQLStoreBase


class TestSQLStore:
    _TEST_PROJECT = "test-model-endpoints"
    _MODEL_ENDPOINT_ID = "some-ep-id"

    @staticmethod
    @pytest.fixture
    def store_connection(tmp_path: Path) -> str:
        return f"sqlite:///{tmp_path / 'test.db'}"

    @classmethod
    @pytest.fixture()
    def _mock_random_endpoint(
        cls,
        state: Optional[str] = None,
    ) -> mlrun.common.schemas.ModelEndpoint:
        def random_labels():
            return {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            }

        return mlrun.common.schemas.ModelEndpoint(
            metadata=mlrun.common.schemas.ModelEndpointMetadata(
                project=cls._TEST_PROJECT,
                labels=random_labels(),
                uid=cls._MODEL_ENDPOINT_ID,
            ),
            spec=mlrun.common.schemas.ModelEndpointSpec(
                function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
                model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
                model_class="classifier",
            ),
            status=mlrun.common.schemas.ModelEndpointStatus(state=state),
        )

    @classmethod
    @pytest.fixture
    def new_sql_store(cls, store_connection: str) -> Iterator[SQLStoreBase]:
        # Generate store object target
        with unittest.mock.patch(
            "mlrun.model_monitoring.helpers.get_connection_string",
            return_value=store_connection,
        ):
            sql_store = cast(
                SQLStoreBase,
                mlrun.model_monitoring.get_store_object(project=cls._TEST_PROJECT),
            )
            yield sql_store
            sql_store.delete_model_endpoints_resources()
            list_of_endpoints = sql_store.list_model_endpoints()
            assert (len(list_of_endpoints)) == 0

    def test_sql_target_list_model_endpoints(
        self,
        new_sql_store: SQLStoreBase,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> None:
        """Testing list model endpoint using SQLStoreBase object. In the following test
        we create two model endpoints and list these endpoints. In addition, this test validates the
        filter optional operation within the list model endpoints API.
        """

        new_sql_store.write_model_endpoint(endpoint=_mock_random_endpoint.flat_dict())

        # Validate that there is a single model endpoint
        list_of_endpoints = new_sql_store.list_model_endpoints()
        assert len(list_of_endpoints) == 1

        # Generate and write the 2nd model endpoint into the DB table
        mock_endpoint_2 = _mock_random_endpoint
        mock_endpoint_2.spec.model = "test_model:latest"
        mock_endpoint_2.spec.function_uri = f"{self._TEST_PROJECT}/function_test"
        mock_endpoint_2.metadata.uid = "12345"
        new_sql_store.write_model_endpoint(endpoint=mock_endpoint_2.flat_dict())

        # Validate that there are exactly two model endpoints within the DB
        list_of_endpoints = new_sql_store.list_model_endpoints()
        assert len(list_of_endpoints) == 2

        # List only the model endpoint that has the model test_model
        filtered_list_of_endpoints = new_sql_store.list_model_endpoints(
            model="test_model"
        )
        assert len(filtered_list_of_endpoints) == 1

        filtered_list_of_endpoints = new_sql_store.list_model_endpoints(
            function="function_test"
        )
        assert len(filtered_list_of_endpoints) == 1

    @staticmethod
    def test_sql_target_patch_endpoint(
        new_sql_store: SQLStoreBase,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> None:
        """Testing the update of a model endpoint using SQLStoreBase object. In the following
        test we update attributes within the model endpoint spec and status and then validate that there
        attributes were actually updated.
        """

        # Generate and write the model endpoint into the DB table
        _mock_random_endpoint.metadata.uid = "1234"
        new_sql_store.write_model_endpoint(_mock_random_endpoint.flat_dict())

        # Generate dictionary of attributes and update the model endpoint
        updated_attributes = {"model": "test_model", "error_count": 2}
        new_sql_store.update_model_endpoint(
            endpoint_id=_mock_random_endpoint.metadata.uid,
            attributes=updated_attributes,
        )

        # Validate that these attributes were actually updated
        endpoint_dict = new_sql_store.get_model_endpoint(
            endpoint_id=_mock_random_endpoint.metadata.uid
        )

        assert endpoint_dict["model"] == "test_model"
        assert endpoint_dict["error_count"] == 2


@pytest.mark.parametrize(
    "time_var",
    [
        datetime.datetime.now(tz=ZoneInfo("Asia/Jerusalem")),
        "2020-05-22T08:59:54.279435+00:00",
    ],
)
def test_convert_to_datetime(time_var: Union[str, datetime.datetime]) -> None:
    time_key = "time"
    event = {time_key: time_var}
    SQLStoreBase._convert_to_datetime(event=event, key=time_key)
    new_time = event[time_key]
    assert isinstance(new_time, datetime.datetime)
    assert new_time.tzinfo == datetime.timezone.utc
