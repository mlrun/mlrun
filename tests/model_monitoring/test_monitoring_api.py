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

import datetime
import pathlib
from typing import Any, Literal
from unittest.mock import Mock, patch

import pytest

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
from mlrun.common.schemas import alert as alert_constants
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpoint,
    ModelEndpointList,
    ModelEndpointMetadata,
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
    ModelEndpointSpec,
    ModelEndpointStatus,
)
from mlrun.common.schemas.notification import Notification, NotificationKind
from mlrun.db import RunDBInterface

from .assets.application import DemoMonitoringApp

APP = "test_app"


def test_read_dataset_as_dataframe():
    # Test list with feature columns
    dataset = [[5.8, 2.8, 5.1, 2.4], [6.0, 2.2, 4.0, 1.0]]
    feature_columns = ["feature_1", "feature_2", "feature_3", "feature_4"]

    df, _ = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset,
        feature_columns=feature_columns,
    )
    assert list(df.columns) == feature_columns
    assert df["feature_1"].to_list() == [dataset[0][0], dataset[1][0]]

    # Test dictionary
    dataset_dict = {}
    for i in range(len(feature_columns)):
        dataset_dict[feature_columns[i]] = [dataset[0][i], dataset[1][i]]
    df, _ = mlrun.model_monitoring.api.read_dataset_as_dataframe(
        dataset=dataset_dict, drop_columns="feature_2"
    )
    feature_columns.remove("feature_2")
    assert list(df.columns) == feature_columns


def test_record_result_updates_last_request() -> None:
    db_mock = Mock(spec=RunDBInterface)
    datetime_mock = datetime.datetime(
        2011, 11, 4, 0, 5, 23, 283000, tzinfo=datetime.timezone.utc
    )
    with patch("mlrun.model_monitoring.api.datetime_now", return_value=datetime_mock):
        with patch("mlrun.model_monitoring.api.mlrun.get_run_db", return_value=db_mock):
            with patch(
                "mlrun.model_monitoring.api.get_or_create_model_endpoint",
                spec=ModelEndpoint,
            ):
                mlrun.model_monitoring.api.record_results(
                    project="some-project",
                    model_path="path/to/model",
                    model_endpoint_name="my-endpoint",
                )

    db_mock.patch_model_endpoint.assert_called_once()
    assert (
        db_mock.patch_model_endpoint.call_args.kwargs["attributes"]["last_request"]
        == datetime_mock
    ), "last_request attribute of the model endpoint was not updated as expected"


def _get_metrics(
    project: str,
    endpoint_ids: list,
    type: Literal["results", "metrics", "all"] = "all",
    events_format: mm_constants.GetEventsFormat = mm_constants.GetEventsFormat.SEPARATION,
):
    results = {
        "mep_id1": [
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.METRIC,
                name="metric-1",
            ),
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.METRIC,
                name="metric-2",
            ),
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.RESULT,
                name="result-a",
            ),
        ],
        "mep_id2": [
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.METRIC,
                name="metric-1",
            ),
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.RESULT,
                name="result-a",
            ),
            ModelEndpointMonitoringMetric(
                project=project,
                app=APP,
                type=ModelEndpointMonitoringMetricType.RESULT,
                name="result-b",
            ),
        ],
    }
    return results


def test_project_create_model_monitoring_alert_configs() -> None:
    db_mock = Mock(spec=RunDBInterface)
    db_mock.get_metrics_by_multiple_endpoints.side_effect = _get_metrics
    project = mlrun.get_or_create_project("mm-project")

    notification = Notification(
        kind=NotificationKind.mail,
        name="my_test_notification",
        email_addresses=["invalid_address@mlrun.com"],
        subject="test alert",
        body="test",
    )
    alert_notification = alert_constants.AlertNotification(
        notification=notification, cooldown_period="5m"
    )

    with patch("mlrun.db.get_run_db", return_value=db_mock):
        mep1 = ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=project.name, uid="mep_id1", name="mep_id1"
            ),
            spec=ModelEndpointSpec(),
            status=ModelEndpointStatus(),
        )
        mep2 = ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=project.name, uid="mep_id2", name="mep_id2"
            ),
            spec=ModelEndpointSpec(),
            status=ModelEndpointStatus(),
        )
        meps_list = ModelEndpointList(endpoints=[mep1, mep2])
        alerts = project.create_model_monitoring_alert_configs(
            name="test",
            endpoints=meps_list,
            summary="summary",
            events=alert_constants.EventKind.FAILED,
            notifications=[alert_notification],
            result_names=[
                f"{APP}.metric-*",
                "*.result-b",
                "mep_id1.test_app.result.metric-3",
            ],
        )
        #  "mep_id1.test_app.result.metric-3" is not exist, but because it is a full result name,
        #  it should raise a warning and create an alert config.
        alert_ids = []
        for alert in alerts:
            alert_ids += alert.entities.ids
        expected_ids = [
            "mep_id1.test_app.result.metric-1",
            "mep_id1.test_app.result.metric-2",
            "mep_id1.test_app.result.metric-3",
            "mep_id2.test_app.result.metric-1",
            "mep_id2.test_app.result.result-b",
        ]
        assert sorted(alert_ids) == sorted(expected_ids)


@pytest.mark.parametrize(
    "function",
    [
        {
            "func": str(pathlib.Path(__file__).parent / "assets" / "application.py"),
            "application_class": DemoMonitoringApp(param_1=1, param_2=2),
        },
        {
            "func": str(pathlib.Path(__file__).parent / "assets" / "application.py"),
            "application_class": "DemoMonitoringApp",
            "param_1": 1,
            "param_2": 2,
        },
    ],
)
def test_create_model_monitoring_function(function: dict[str, Any]) -> None:
    app = mlrun.model_monitoring.api._create_model_monitoring_function_base(
        project="", name="my-app", **function
    )
    assert app.metadata.name == "my-app"

    steps = app.spec.graph.steps

    assert "PrepareMonitoringEvent" in app.spec.graph.steps
    assert "DemoMonitoringApp" in app.spec.graph.steps
    assert "PushToMonitoringWriter" in app.spec.graph.steps
    assert "ApplicationErrorHandler" in app.spec.graph.steps

    app_step = steps["DemoMonitoringApp"]
    assert app_step.class_args == {"param_1": 1, "param_2": 2}

    with pytest.raises(NotImplementedError):
        app.to_mock_server()
