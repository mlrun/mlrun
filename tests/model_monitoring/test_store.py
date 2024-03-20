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
#

import string
import time
import typing
from random import choice, randint
from typing import Optional

import pytest
from _pytest.fixtures import FixtureRequest

import mlrun.common.schemas
import mlrun.model_monitoring
import mlrun.model_monitoring.db.stores.sqldb.sql_store
from mlrun.common.schemas.model_monitoring import SchedulingKeys, WriterEvent
from mlrun.model_monitoring.db.stores import (  # noqa: F401
    StoreBase,
)
from mlrun.model_monitoring.writer import _AppResultEvent

SQLstoreObject = typing.TypeVar("SQLstoreObject", bound="StoreBase")


TEST_PROJECT = "test_model_endpoints"
STORE_CONNECTION = "sqlite:///test.db"


def _mock_random_endpoint(
    state: Optional[str] = None,
) -> mlrun.common.schemas.ModelEndpoint:
    def random_labels():
        return {f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)}

    return mlrun.common.schemas.ModelEndpoint(
        metadata=mlrun.common.schemas.ModelEndpointMetadata(
            project=TEST_PROJECT, labels=random_labels(), uid="some-ep-id"
        ),
        spec=mlrun.common.schemas.ModelEndpointSpec(
            function_uri=f"test/function_{randint(0, 100)}:v{randint(0, 100)}",
            model=f"model_{randint(0, 100)}:v{randint(0, 100)}",
            model_class="classifier",
        ),
        status=mlrun.common.schemas.ModelEndpointStatus(state=state),
    )


@pytest.fixture(params=[0])
def event(request: FixtureRequest) -> _AppResultEvent:
    return _AppResultEvent(
        {
            WriterEvent.ENDPOINT_ID: "some-ep-id",
            WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
            WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
            WriterEvent.APPLICATION_NAME: "dummy-app",
            WriterEvent.RESULT_NAME: "data-drift-0",
            WriterEvent.RESULT_KIND: 0,
            WriterEvent.RESULT_VALUE: 0.32,
            WriterEvent.RESULT_STATUS: request.param,
            WriterEvent.RESULT_EXTRA_DATA: "",
        }
    )


def test_sql_write_application_result(event: _AppResultEvent):
    # Generate store object target
    store_type_object = mlrun.model_monitoring.db.ObjectStoreType(value="sql")
    sql_store: SQLstoreObject = store_type_object.to_object_store(
        project=TEST_PROJECT, store_connection=STORE_CONNECTION
    )

    sql_store._create_tables_if_not_exist()

    # Generate mock model endpoint
    mock_endpoint_1 = _mock_random_endpoint()

    # Validate that there are no model endpoints or application result records at the moment
    _clean_model_endpoint_from_db(
        store=sql_store, endpoint_id=mock_endpoint_1.metadata.uid
    )
    sql_store.delete_application_result(
        endpoint_id=mock_endpoint_1.metadata.uid,
        application_name=event[WriterEvent.APPLICATION_NAME],
    )

    # Generate mock model endpoint
    sql_store.write_model_endpoint(endpoint=mock_endpoint_1.flat_dict())

    # Write a dummy application result event
    sql_store.write_application_result(event=event)

    application_filter_dict = sql_store.filter_endpoint_and_application_name(
        endpoint_id=event[WriterEvent.ENDPOINT_ID],
        application_name=event[WriterEvent.APPLICATION_NAME],
    )

    application_record = sql_store._get(
        table=sql_store.ApplicationResultsTable, **application_filter_dict
    )

    assert application_record.endpoint_id == event[WriterEvent.ENDPOINT_ID]
    assert application_record.application_name == event[WriterEvent.APPLICATION_NAME]

    assert (
        application_record.result_value
        == event[mlrun.common.schemas.model_monitoring.WriterEvent.RESULT_VALUE]
    )

    # Clean resources
    sql_store.delete_application_result(
        endpoint_id=mock_endpoint_1.metadata.uid,
        application_name=application_record.application_name,
    )
    _clean_model_endpoint_from_db(
        store=sql_store, endpoint_id=mock_endpoint_1.metadata.uid
    )


def test_sql_last_analyzed_result(event: _AppResultEvent):
    # Generate store object target
    store_type_object = mlrun.model_monitoring.db.ObjectStoreType(value="sql")
    sql_store: SQLstoreObject = store_type_object.to_object_store(
        project=TEST_PROJECT, store_connection=STORE_CONNECTION
    )

    sql_store._create_tables_if_not_exist()

    # Generate mock model endpoint
    mock_endpoint_1 = _mock_random_endpoint()

    # Validate that there are no model endpoints  or last analyzed records at the moment
    _clean_model_endpoint_from_db(
        store=sql_store, endpoint_id=mock_endpoint_1.metadata.uid
    )
    sql_store.delete_last_analyzed(mock_endpoint_1.metadata.uid)

    # Write mock model endpoint to DB
    sql_store.write_model_endpoint(endpoint=mock_endpoint_1.flat_dict())

    # Try to get last analyzed value, we expect it to be empty
    last_analyzed = sql_store.get_last_analyzed(
        endpoint_id=mock_endpoint_1.metadata.uid,
        application_name=event[WriterEvent.APPLICATION_NAME],
    )
    assert not last_analyzed

    # Let's ingest a dummy epoch time record and validate it has been stored as expected
    epoch_time = int(time.time())
    attributes = {SchedulingKeys.LAST_ANALYZED: epoch_time}
    sql_store.update_last_analyzed(
        endpoint_id=mock_endpoint_1.metadata.uid,
        application_name=event[WriterEvent.APPLICATION_NAME],
        attributes=attributes,
    )

    last_analyzed = sql_store.get_last_analyzed(
        endpoint_id=mock_endpoint_1.metadata.uid,
        application_name=event[WriterEvent.APPLICATION_NAME],
    )

    assert last_analyzed == epoch_time

    # Clean resources
    sql_store.delete_last_analyzed(mock_endpoint_1.metadata.uid)
    _clean_model_endpoint_from_db(
        store=sql_store, endpoint_id=mock_endpoint_1.metadata.uid
    )


def _clean_model_endpoint_from_db(store, endpoint_id):
    store.delete_model_endpoint(endpoint_id)
    list_of_endpoints = store.list_model_endpoints(uids=[endpoint_id])
    assert (len(list_of_endpoints)) == 0
