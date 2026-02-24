# Copyright 2026 Iguazio
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

import time
import unittest.mock
from datetime import UTC, datetime, timedelta

import pytest

import mlrun
import mlrun.common.schemas.notification
import mlrun.db.httpdb
import mlrun.model_monitoring
from mlrun.common.schemas.alert import (
    AlertCriteria,
    AlertSeverity,
    AlertTrigger,
    EventEntities,
    EventEntityKind,
    EventKind,
    ResetPolicy,
    _event_kind_entity_map,
)
from mlrun.common.schemas.model_monitoring.constants import (
    MonitoringAlertNames,
    WriterEvent,
    WriterEventKind,
)
from mlrun.datastore.datastore_profile import DatastoreProfilePostgreSQL
from mlrun.model_monitoring.writer import (
    KindChoice,
    WriterGraphFactory,
    WriterLagEventsGenerator,
)
from mlrun.serving.server import GraphContext
from mlrun.serving.states import TaskStep

TEST_PROJECT = "test-lag-detection"


# -- Schema tests --


class TestAlertSchemaEnums:
    def test_model_monitoring_infra_in_event_entity_kind(self):
        assert EventEntityKind.MODEL_MONITORING_INFRA == "model-monitoring-infra"

    def test_model_monitoring_lag_detected_in_event_kind(self):
        assert (
            EventKind.MODEL_MONITORING_LAG_DETECTED == "model-monitoring-lag-detected"
        )

    def test_lag_detected_mapped_to_infra_entity(self):
        assert _event_kind_entity_map[EventKind.MODEL_MONITORING_LAG_DETECTED] == [
            EventEntityKind.MODEL_MONITORING_INFRA
        ]


# -- WriterLagEventsGenerator tests --


class TestWriterLagEventsGenerator:
    @staticmethod
    def _make_event(
        end_infer_time: datetime,
        endpoint_name: str = "my-endpoint",
        endpoint_id: str = "ep-123",
        application_name: str = "my-app",
    ) -> dict:
        return {
            WriterEvent.END_INFER_TIME: end_infer_time,
            WriterEvent.ENDPOINT_NAME: endpoint_name,
            WriterEvent.ENDPOINT_ID: endpoint_id,
            WriterEvent.APPLICATION_NAME: application_name,
        }

    def test_generates_event_when_lag_exceeds_threshold(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,  # 5 min
            lag_event_cooldown_seconds=0,
        )
        step.context = unittest.mock.Mock(worker_id=3)
        old_time = datetime.now(tz=UTC) - timedelta(minutes=10)
        event = self._make_event(end_infer_time=old_time)

        result = step.do(event)

        assert result["value_dict"].pop("lag_seconds") >= 300
        assert result == {
            "kind": "model-monitoring-lag-detected",
            "timestamp": None,
            "entity": {
                "kind": "model-monitoring-infra",
                "project": TEST_PROJECT,
                "ids": [f"{TEST_PROJECT}.writer.3"],
            },
            "value_dict": {
                "worker_id": 3,
                "endpoint_name": "my-endpoint",
                "endpoint_id": "ep-123",
                "app_name": "my-app",
            },
        }

    def test_returns_none_when_disabled(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
        )
        old_time = datetime.now(tz=UTC) - timedelta(minutes=10)
        event = self._make_event(end_infer_time=old_time)

        assert step.do(event) is None

    def test_returns_none_when_lag_below_threshold(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,
            lag_event_cooldown_seconds=0,
        )
        recent_time = datetime.now(tz=UTC) - timedelta(seconds=30)

        assert step.do(self._make_event(end_infer_time=recent_time)) is None

    def test_returns_none_when_end_infer_time_missing(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,
            lag_event_cooldown_seconds=0,
        )
        event = {
            WriterEvent.ENDPOINT_NAME: "my-endpoint",
            WriterEvent.ENDPOINT_ID: "ep-123",
        }

        assert step.do(event) is None

    def test_cooldown_is_per_worker(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,
            lag_event_cooldown_seconds=9999,  # very long cooldown
        )
        step.context = unittest.mock.Mock(worker_id=0)
        old_time = datetime.now(tz=UTC) - timedelta(minutes=10)

        # Worker 0 triggers, then gets blocked by cooldown
        first_w0 = step.do(self._make_event(end_infer_time=old_time))
        assert first_w0["kind"] == "model-monitoring-lag-detected"
        assert step.do(self._make_event(end_infer_time=old_time)) is None

        # Worker 1 can still trigger — independent cooldown
        step.context.worker_id = 1
        first_w1 = step.do(self._make_event(end_infer_time=old_time))
        assert first_w1["kind"] == "model-monitoring-lag-detected"

    def test_cooldown_expires_allows_new_event(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,
            lag_event_cooldown_seconds=0.1,  # 100ms cooldown
        )
        old_time = datetime.now(tz=UTC) - timedelta(minutes=10)

        first = step.do(self._make_event(end_infer_time=old_time))
        time.sleep(0.15)
        second = step.do(self._make_event(end_infer_time=old_time))

        assert first["kind"] == "model-monitoring-lag-detected"
        assert second["kind"] == "model-monitoring-lag-detected"

    def test_entity_id_defaults_worker_0_when_no_context(self):
        step = WriterLagEventsGenerator(
            project="my-project",
            lag_threshold_seconds=60,
            lag_event_cooldown_seconds=0,
        )
        old_time = datetime.now(tz=UTC) - timedelta(minutes=5)

        result = step.do(self._make_event(end_infer_time=old_time))

        assert result["entity"]["ids"] == ["my-project.writer.0"]

    def test_handles_string_end_infer_time(self):
        step = WriterLagEventsGenerator(
            project=TEST_PROJECT,
            lag_threshold_seconds=300,
            lag_event_cooldown_seconds=0,
        )
        old_time = (datetime.now(tz=UTC) - timedelta(minutes=10)).isoformat()
        event = self._make_event(end_infer_time=old_time)

        result = step.do(event)

        assert result["kind"] == "model-monitoring-lag-detected"

    def test_context_worker_id_propagates_via_task_step(self):
        """Verify worker_id flows from GraphContext through TaskStep to storey step.

        Note: Accessing task_step._object is necessary here to verify the internal
        wiring between TaskStep and the underlying storey class. This tests the
        serving framework's context propagation mechanism.
        """
        task_step = TaskStep(
            class_name="mlrun.model_monitoring.writer.WriterLagEventsGenerator",
            class_args={
                "project": TEST_PROJECT,
                "lag_threshold_seconds": 60,
                "lag_event_cooldown_seconds": 0,
            },
        )

        nuclio_ctx = unittest.mock.Mock(worker_id=7)
        graph_context = GraphContext(nuclio_context=nuclio_ctx)
        task_step.init_object(context=graph_context, namespace={}, mode="sync")

        assert task_step._object.context.worker_id == 7


# -- KindChoice tests --


class TestKindChoiceRouting:
    def test_metric_routes_to_lag(self):
        choice = KindChoice()
        event = {"kind": WriterEventKind.METRIC}

        outlets = choice.select_outlets(event)

        assert outlets == ["tsdb_metrics", "lag_events_generator"]

    def test_result_routes_to_lag_and_alert(self):
        choice = KindChoice()
        event = {"kind": WriterEventKind.RESULT}

        outlets = choice.select_outlets(event)

        assert outlets == [
            "tsdb_app_results",
            "alert_generator",
            "lag_events_generator",
        ]

    def test_stats_does_not_route_to_lag(self):
        choice = KindChoice()
        event = {"kind": WriterEventKind.STATS}

        outlets = choice.select_outlets(event)

        assert outlets == ["stats_writer"]


# -- Writer graph topology tests --


class TestWriterGraphWithLag:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(mlrun.mlconf, "system_id", "123456")

    @staticmethod
    def _build_graph(
        lag_threshold_minutes: int | None = None,
        lag_event_cooldown_minutes: int | None = None,
    ):
        project_name = "test-graph"
        project = mlrun.get_or_create_project(project_name, allow_cross_project=True)
        fn = project.set_function(kind="serving", name="writer-fn")

        tsdb_profile = DatastoreProfilePostgreSQL(
            name="tsdb-test",
            user="test",
            password="test",
            host="localhost",
            port=5432,
        )
        tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=project_name, profile=tsdb_profile
        )

        factory = WriterGraphFactory(
            parquet_path="/tmp/test",
            lag_threshold_minutes=lag_threshold_minutes,
            lag_event_cooldown_minutes=lag_event_cooldown_minutes,
        )
        factory.apply_writer_graph(fn, tsdb_connector)
        return fn.spec.graph

    def test_lag_step_feeds_into_filter_none(self):
        graph = self._build_graph(
            lag_threshold_minutes=10, lag_event_cooldown_minutes=5
        )
        filter_step = graph["filter_none"]

        assert set(filter_step.after) == {"alert_generator", "lag_events_generator"}

    def test_lag_step_receives_from_kind_choice(self):
        graph = self._build_graph()
        lag_step = graph["lag_events_generator"]

        assert lag_step.after == ["kind_choice_step"]


# -- Config tests --


class TestLagDetectionConfig:
    def test_config_defaults_exist(self):
        lag_cfg = mlrun.mlconf.model_endpoint_monitoring.lag_detection
        assert int(lag_cfg.min_lag_threshold_minutes) == 5
        assert int(lag_cfg.default_lag_threshold_minutes) == 60
        assert int(lag_cfg.default_lag_event_cooldown_minutes) == 30

    @pytest.mark.parametrize(
        "base_period, expected_threshold, expected_cooldown",
        [
            # base_period smaller than config defaults -> clamped to base_period
            (10, 10, 5),
            # base_period larger than config defaults -> uses config defaults
            (120, 60, 30),
            # base_period equals config default_lag_threshold -> uses config values
            (60, 60, 30),
        ],
    )
    def test_default_lag_values_from_config_and_base_period(
        self, base_period, expected_threshold, expected_cooldown
    ):
        """Verify: threshold = min(config, base_period),
        cooldown = min(config, base_period // 2)."""
        lag_cfg = mlrun.mlconf.model_endpoint_monitoring.lag_detection
        config_threshold = int(lag_cfg.default_lag_threshold_minutes)
        config_cooldown = int(lag_cfg.default_lag_event_cooldown_minutes)

        computed_threshold = min(config_threshold, base_period)
        computed_cooldown = min(config_cooldown, base_period // 2)

        assert computed_threshold == expected_threshold
        assert computed_cooldown == expected_cooldown


# -- Parameter chain tests (ML-12079) --


class TestEnableModelMonitoringLagValidation:
    @staticmethod
    @pytest.fixture()
    def mock_db():
        mock = unittest.mock.Mock()
        with unittest.mock.patch("mlrun.db.get_run_db", return_value=mock):
            yield mock

    @staticmethod
    @pytest.fixture()
    def project() -> mlrun.projects.MlrunProject:
        return unittest.mock.Mock()

    def test_lag_params_forwarded_to_db(self, project, mock_db):
        lag_threshold = 15
        lag_event_cooldown = 7

        mlrun.projects.MlrunProject.enable_model_monitoring(
            project,
            deploy_histogram_data_drift_app=False,
            lag_threshold=lag_threshold,
            lag_event_cooldown=lag_event_cooldown,
        )

        call_kwargs = mock_db.enable_model_monitoring.call_args.kwargs
        assert call_kwargs["lag_threshold"] == lag_threshold
        assert call_kwargs["lag_event_cooldown"] == lag_event_cooldown

    def test_lag_params_default_none_forwarded_to_db(self, project, mock_db):
        mlrun.projects.MlrunProject.enable_model_monitoring(
            project,
            deploy_histogram_data_drift_app=False,
        )

        call_kwargs = mock_db.enable_model_monitoring.call_args.kwargs
        assert call_kwargs["lag_threshold"] is None
        assert call_kwargs["lag_event_cooldown"] is None


class TestHTTPDBLagParams:
    def test_lag_params_added_to_query_when_set(self):
        lag_threshold = 15
        lag_event_cooldown = 7
        db = mlrun.db.httpdb.HTTPRunDB("http://fake")
        db.api_call = unittest.mock.Mock()

        db.enable_model_monitoring(
            project="test",
            lag_threshold=lag_threshold,
            lag_event_cooldown=lag_event_cooldown,
        )

        call_kwargs = db.api_call.call_args.kwargs
        assert call_kwargs["params"]["lag_threshold"] == lag_threshold
        assert call_kwargs["params"]["lag_event_cooldown"] == lag_event_cooldown

    def test_lag_params_omitted_from_query_when_none(self):
        db = mlrun.db.httpdb.HTTPRunDB("http://fake")
        db.api_call = unittest.mock.Mock()

        db.enable_model_monitoring(project="test")

        params = db.api_call.call_args.kwargs["params"]
        assert "lag_threshold" not in params
        assert "lag_event_cooldown" not in params


# -- SDK lag alert methods tests (ML-11675) --

_LAG_ALERT_PROJECT = "test-lag-alert"


@pytest.fixture()
def lag_alert_mock_db():
    mock = unittest.mock.Mock()
    with unittest.mock.patch("mlrun.db.get_run_db", return_value=mock):
        yield mock


@pytest.fixture()
def lag_alert_project():
    mock = unittest.mock.Mock()
    mock.name = _LAG_ALERT_PROJECT
    return mock


def _make_notification(name: str = "test-notif"):
    return mlrun.common.schemas.notification.Notification(
        kind="slack",
        name=name,
        secret_params={"webhook": "https://hooks.slack.com/test"},
    )


class TestSetModelMonitoringLagAlert:
    def test_creates_single_wildcard_alert(self, lag_alert_mock_db, lag_alert_project):
        mlrun.projects.MlrunProject.set_model_monitoring_lag_alert(
            lag_alert_project,
            notifications=_make_notification(),
        )

        lag_alert_mock_db.store_alert_config.assert_called_once()
        call_args = lag_alert_mock_db.store_alert_config.call_args
        assert call_args.args[0] == MonitoringAlertNames.LAG_DETECTED
        assert call_args.args[1].entities.ids == ["*"]

    def test_alert_config_fields(self, lag_alert_mock_db, lag_alert_project):
        period = "10m"
        count = 2

        mlrun.projects.MlrunProject.set_model_monitoring_lag_alert(
            lag_alert_project,
            notifications=_make_notification(),
            period=period,
            count=count,
        )

        alert_data = lag_alert_mock_db.store_alert_config.call_args.args[1]
        assert alert_data.severity == AlertSeverity.MEDIUM
        assert alert_data.reset_policy == ResetPolicy.AUTO
        assert alert_data.trigger == AlertTrigger(
            events=[EventKind.MODEL_MONITORING_LAG_DETECTED]
        )
        assert alert_data.criteria == AlertCriteria(count=count, period=period)
        assert alert_data.entities == EventEntities(
            kind=EventEntityKind.MODEL_MONITORING_INFRA,
            project=_LAG_ALERT_PROJECT,
            ids=["*"],
        )

    def test_wraps_single_notification_in_list(
        self, lag_alert_mock_db, lag_alert_project
    ):
        notification = _make_notification()

        mlrun.projects.MlrunProject.set_model_monitoring_lag_alert(
            lag_alert_project,
            notifications=notification,
        )

        alert_data = lag_alert_mock_db.store_alert_config.call_args.args[1]
        assert len(alert_data.notifications) == 1
        assert alert_data.notifications[0].notification.name == notification.name

    def test_accepts_list_of_notifications(self, lag_alert_mock_db, lag_alert_project):
        notifications = [
            _make_notification("n1"),
            _make_notification("n2"),
        ]

        mlrun.projects.MlrunProject.set_model_monitoring_lag_alert(
            lag_alert_project,
            notifications=notifications,
        )

        alert_data = lag_alert_mock_db.store_alert_config.call_args.args[1]
        assert len(alert_data.notifications) == len(notifications)
        actual_names = [n.notification.name for n in alert_data.notifications]
        assert actual_names == ["n1", "n2"]


class TestDeleteModelMonitoringLagAlert:
    def test_deletes_alert(self, lag_alert_mock_db, lag_alert_project):
        mlrun.projects.MlrunProject.delete_model_monitoring_lag_alert(
            lag_alert_project,
        )

        lag_alert_mock_db.delete_alert_config.assert_called_once_with(
            MonitoringAlertNames.LAG_DETECTED,
            project_name=_LAG_ALERT_PROJECT,
        )

    def test_ignores_not_found_errors(self, lag_alert_mock_db, lag_alert_project):
        lag_alert_mock_db.delete_alert_config.side_effect = (
            mlrun.errors.MLRunNotFoundError("not found")
        )

        mlrun.projects.MlrunProject.delete_model_monitoring_lag_alert(
            lag_alert_project,
        )

        lag_alert_mock_db.delete_alert_config.assert_called_once()
