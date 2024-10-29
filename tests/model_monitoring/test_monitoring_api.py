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
from unittest.mock import Mock, patch

import pytest

import mlrun.model_monitoring.api
from mlrun.db import RunDBInterface
from mlrun.model_monitoring import ModelEndpoint

from .assets.application import DemoMonitoringApp


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
        == datetime_mock.isoformat()
    ), "last_request attribute of the model endpoint was not updated as expected"


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
def test_create_model_monitoring_function(function) -> None:
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
