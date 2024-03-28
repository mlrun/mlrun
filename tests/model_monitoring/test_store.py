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
#

import string
import time
import typing
import unittest.mock
from random import choice, randint
from typing import Optional

import pytest

import mlrun.common.schemas
import mlrun.model_monitoring
import mlrun.model_monitoring.db.stores.sqldb.sql_store
from mlrun.common.schemas.model_monitoring import WriterEvent
from mlrun.model_monitoring.db.stores import (  # noqa: F401
    StoreBase,
)
from mlrun.model_monitoring.writer import _AppResultEvent

SQLstoreObject = typing.TypeVar("SQLstoreObject", bound="StoreBase")


class TestSQLStore:
    _TEST_PROJECT = "test_model_endpoints"
    _MODEL_ENDPOINT_ID = "some-ep-id"
    _STORE_CONNECTION = "sqlite:///test.db"

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

    @staticmethod
    @pytest.fixture
    def event(
        request: pytest.FixtureRequest,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> _AppResultEvent:
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: _mock_random_endpoint.metadata.uid,
                WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.RESULT_NAME: "data-drift-0",
                WriterEvent.RESULT_KIND: 0,
                WriterEvent.RESULT_VALUE: 0.32,
                WriterEvent.RESULT_STATUS: 0,
                WriterEvent.RESULT_EXTRA_DATA: "",
            }
        )

    @staticmethod
    @pytest.fixture
    def event_v2(
        request: pytest.FixtureRequest,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> _AppResultEvent:
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: _mock_random_endpoint.metadata.uid,
                WriterEvent.START_INFER_TIME: "2023-09-20 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-20 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                WriterEvent.RESULT_NAME: "data-drift-0",
                WriterEvent.RESULT_KIND: 1,
                WriterEvent.RESULT_VALUE: 5.15,
                WriterEvent.RESULT_STATUS: 1,
                WriterEvent.RESULT_EXTRA_DATA: "",
            }
        )

    @staticmethod
    @pytest.fixture(autouse=True)
    def init_sql_tables(new_sql_store: SQLstoreObject):
        new_sql_store._create_tables_if_not_exist()

    @classmethod
    @pytest.fixture
    def new_sql_store(cls) -> SQLstoreObject:
        # Generate store object target
        store_type_object = mlrun.model_monitoring.db.ObjectStoreFactory(value="sql")
        with unittest.mock.patch(
            "mlrun.model_monitoring.helpers.get_connection_string",
            return_value=cls._STORE_CONNECTION,
        ):
            sql_store: SQLstoreObject = store_type_object.to_object_store(
                project=cls._TEST_PROJECT
            )
            yield sql_store
            list_of_endpoints = sql_store.list_model_endpoints()
            sql_store.delete_model_endpoints_resources(list_of_endpoints)
            list_of_endpoints = sql_store.list_model_endpoints()
            assert (len(list_of_endpoints)) == 0

    def test_sql_write_application_result(
        cls,
        event: _AppResultEvent,
        event_v2: _AppResultEvent,
        new_sql_store: SQLstoreObject,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        # Generate mock model endpoint
        new_sql_store.write_model_endpoint(endpoint=_mock_random_endpoint.flat_dict())

        # Write a dummy application result event
        new_sql_store.write_application_result(event=event)

        cls.assert_application_record(event=event, new_sql_store=new_sql_store)

        # Write a 2nd application result event - we expect it to overwrite the existing record
        new_sql_store.write_application_result(event=event_v2)

        cls.assert_application_record(event=event_v2, new_sql_store=new_sql_store)

    @staticmethod
    def assert_application_record(
        event: _AppResultEvent, new_sql_store: SQLstoreObject
    ):
        application_filter_dict = new_sql_store.filter_endpoint_and_application_name(
            endpoint_id=event[WriterEvent.ENDPOINT_ID],
            application_name=event[WriterEvent.APPLICATION_NAME],
        )

        application_record = new_sql_store._get(
            table=new_sql_store.ApplicationResultsTable, **application_filter_dict
        )

        assert application_record.endpoint_id == event[WriterEvent.ENDPOINT_ID]
        assert (
            application_record.application_name == event[WriterEvent.APPLICATION_NAME]
        )

        assert (
            application_record.result_value
            == event[mlrun.common.schemas.model_monitoring.WriterEvent.RESULT_VALUE]
        )

        assert application_record.uid == new_sql_store._generate_application_result_uid(
            event=event
        )

    @staticmethod
    def test_sql_last_analyzed_result(
        event: _AppResultEvent,
        new_sql_store: SQLstoreObject,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        # Write mock model endpoint to DB
        new_sql_store.write_model_endpoint(endpoint=_mock_random_endpoint.flat_dict())

        # Try to get last analyzed value, we expect it to be empty
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            new_sql_store.get_last_analyzed(
                endpoint_id=_mock_random_endpoint.metadata.uid,
                application_name=event[WriterEvent.APPLICATION_NAME],
            )

        # Let's ingest a dummy epoch time record and validate it has been stored as expected
        epoch_time = int(time.time())
        new_sql_store.update_last_analyzed(
            endpoint_id=_mock_random_endpoint.metadata.uid,
            application_name=event[WriterEvent.APPLICATION_NAME],
            last_analyzed=epoch_time,
        )

        last_analyzed = new_sql_store.get_last_analyzed(
            endpoint_id=_mock_random_endpoint.metadata.uid,
            application_name=event[WriterEvent.APPLICATION_NAME],
        )

        assert last_analyzed == epoch_time
