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
import time
import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from random import choice, randint
from typing import Optional, Union, cast
from zoneinfo import ZoneInfo

import pytest

import mlrun.common.schemas
import mlrun.model_monitoring
from mlrun.common.schemas.model_monitoring import (
    MetricData,
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
    ProjectSecretKeys,
    ResultData,
    WriterEvent,
    WriterEventKind,
)
from mlrun.model_monitoring.db.stores.sqldb import models
from mlrun.model_monitoring.db.stores.sqldb.sql_store import SQLStoreBase
from mlrun.model_monitoring.writer import _AppResultEvent


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

    @staticmethod
    @pytest.fixture
    def event(
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> _AppResultEvent:
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: _mock_random_endpoint.metadata.uid,
                WriterEvent.START_INFER_TIME: "2023-09-19 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-19 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                ResultData.RESULT_NAME: "data-drift-0",
                ResultData.RESULT_KIND: 0,
                ResultData.RESULT_VALUE: 0.32,
                ResultData.RESULT_STATUS: 0,
                ResultData.RESULT_EXTRA_DATA: "",
            }
        )

    @staticmethod
    @pytest.fixture
    def event_v2(
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> _AppResultEvent:
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: _mock_random_endpoint.metadata.uid,
                WriterEvent.START_INFER_TIME: "2023-09-20 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-20 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "dummy-app",
                ResultData.RESULT_NAME: "data-drift-0",
                ResultData.RESULT_KIND: 1,
                ResultData.RESULT_VALUE: 5.15,
                ResultData.RESULT_STATUS: 1,
                ResultData.RESULT_EXTRA_DATA: "",
            }
        )

    @staticmethod
    @pytest.fixture
    def metric_event(
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> _AppResultEvent:
        return _AppResultEvent(
            {
                WriterEvent.ENDPOINT_ID: _mock_random_endpoint.metadata.uid,
                WriterEvent.START_INFER_TIME: "2023-09-22 14:26:06.501084",
                WriterEvent.END_INFER_TIME: "2023-09-22 16:26:06.501084",
                WriterEvent.APPLICATION_NAME: "smart-app",
                MetricData.METRIC_NAME: "met-metric",
                MetricData.METRIC_VALUE: 0.4,
            }
        )

    @staticmethod
    @pytest.fixture(autouse=True)
    def init_sql_tables(new_sql_store: SQLStoreBase) -> None:
        new_sql_store.create_tables()

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

    @classmethod
    def test_sql_write_application_event(
        cls,
        event: _AppResultEvent,
        event_v2: _AppResultEvent,
        new_sql_store: SQLStoreBase,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ):
        # Generate mock model endpoint
        new_sql_store.write_model_endpoint(endpoint=_mock_random_endpoint.flat_dict())

        # Write a dummy application result event
        new_sql_store.write_application_event(event=event)

        cls.assert_application_record(event=event, new_sql_store=new_sql_store)

        # Write a 2nd application result event - we expect it to overwrite the existing record
        new_sql_store.write_application_event(event=event_v2)

        cls.assert_application_record(event=event_v2, new_sql_store=new_sql_store)

    @staticmethod
    def assert_application_record(event: _AppResultEvent, new_sql_store: SQLStoreBase):
        criteria = [
            new_sql_store.application_results_table.endpoint_id
            == event[WriterEvent.ENDPOINT_ID],
            new_sql_store.application_results_table.application_name
            == event[WriterEvent.APPLICATION_NAME],
        ]

        application_record = new_sql_store._get(
            table=new_sql_store.application_results_table, criteria=criteria
        )

        assert application_record.endpoint_id == event[WriterEvent.ENDPOINT_ID]
        assert (
            application_record.application_name == event[WriterEvent.APPLICATION_NAME]
        )

        assert application_record.result_value == event[ResultData.RESULT_VALUE]

        assert application_record.uid == new_sql_store._generate_application_result_uid(
            event=event
        )

    @staticmethod
    def test_sql_last_analyzed_result(
        event: _AppResultEvent,
        new_sql_store: SQLStoreBase,
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

    @classmethod
    def test_get_metrics(
        cls,
        new_sql_store: SQLStoreBase,
        metric_event: _AppResultEvent,
        _mock_random_endpoint: mlrun.common.schemas.ModelEndpoint,
    ) -> None:
        def get_metrics() -> list[ModelEndpointMonitoringMetric]:
            return new_sql_store.get_model_endpoint_metrics(
                endpoint_id=cls._MODEL_ENDPOINT_ID,
                type=ModelEndpointMonitoringMetricType.METRIC,
            )

        new_sql_store.write_model_endpoint(endpoint=_mock_random_endpoint.flat_dict())
        new_sql_store.write_application_event(
            event=metric_event, kind=WriterEventKind.METRIC
        )
        metrics = get_metrics()
        assert len(metrics) == 1, "The metrics number is wrong"
        assert (
            metrics[0].full_name == f"{cls._TEST_PROJECT}.smart-app.metric.met-metric"
        ), "The metric FQN is different than expected"
        new_sql_store._delete_application_metrics(
            endpoint_id=cls._MODEL_ENDPOINT_ID, application_name="smart-app"
        )
        assert get_metrics() == [], "Metric remained after deletion"


class TestMonitoringSchedules:
    @staticmethod
    @pytest.fixture
    def in_mem_connection() -> str:
        return "sqlite://"

    @staticmethod
    @pytest.fixture
    def sqlite_store(
        in_mem_connection: str, monkeypatch: pytest.MonkeyPatch
    ) -> SQLStoreBase:
        with monkeypatch.context() as mp_ctx:
            mp_ctx.setenv(
                ProjectSecretKeys.ENDPOINT_STORE_CONNECTION, in_mem_connection
            )
            store = mlrun.model_monitoring.get_store_object(project="tmp_proj")
        store._create_tables_if_not_exist()
        return store

    @staticmethod
    def test_unique_last_analyzed_per_app(sqlite_store: SQLStoreBase) -> None:
        endpoint_id = "ep-abc123"
        app1_name = "app-A"
        app1_last_analyzed = 1716720842
        app2_name = "app-B"

        sqlite_store.update_last_analyzed(
            endpoint_id=endpoint_id,
            application_name=app1_name,
            last_analyzed=app1_last_analyzed,
        )

        assert (
            sqlite_store.get_last_analyzed(
                endpoint_id=endpoint_id, application_name=app1_name
            )
            == app1_last_analyzed
        )

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            sqlite_store.get_last_analyzed(
                endpoint_id=endpoint_id, application_name=app2_name
            )


@pytest.mark.parametrize(
    ("connection_string", "expected_table"),
    [
        (None, models.SQLiteApplicationResultTable),
        ("sqlite://", models.SQLiteApplicationResultTable),
        (
            "mysql+pymysql://<username>:<password>@<host>/<dbname>",
            models.MySQLApplicationResultTable,
        ),
    ],
)
def test_get_app_metrics_table(
    connection_string: Optional[str], expected_table: type
) -> None:
    assert (
        models._get_application_result_table(connection_string=connection_string)
        == expected_table
    ), "The metrics table is different than expected"


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime.now(tz=ZoneInfo("Asia/Jerusalem")),
        "2020-05-22T08:59:54.279435+00:00",
    ],
)
def test_convert_to_datetime(time: Union[str, datetime.datetime]) -> None:
    time_key = "time"
    event = {time_key: time}
    SQLStoreBase._convert_to_datetime(event=event, key=time_key)
    new_time = event[time_key]
    assert isinstance(new_time, datetime.datetime)
    assert new_time.tzinfo == datetime.timezone.utc
