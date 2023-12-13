# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from mlrun.model_monitoring.llm_application import LLMMonitoringApp
from mlrun.model_monitoring.metrics import LLMEvaluateMetric, LLMJudgeBaseMetric


@pytest.fixture
def mock_metrics():
    return [MagicMock(spec=LLMEvaluateMetric), MagicMock(spec=LLMJudgeBaseMetric)]


@pytest.fixture
def mock_dataframe():
    return pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})


@pytest.fixture
def llm_monitoring_app(mock_metrics):
    return LLMMonitoringApp(name="TestApp", metrics=mock_metrics)


def test_initialization(mock_metrics):
    app = LLMMonitoringApp(name="TestApp", metrics=mock_metrics)
    assert app.name == "TestApp"
    assert app.metrics == mock_metrics


def test_metrics_getter_setter(llm_monitoring_app, mock_metrics):
    new_metrics = [MagicMock(spec=LLMEvaluateMetric)]
    llm_monitoring_app.metrics = new_metrics
    assert llm_monitoring_app.metrics == new_metrics


def test_compute_metrics_over_data(llm_monitoring_app, mock_dataframe):
    # Assuming compute_metrics_over_data returns a dictionary of metric results
    metrics_names = ["metric1", "metric2"]
    results = llm_monitoring_app.compute_metrics_over_data(
        mock_dataframe, mock_dataframe, metrics_names
    )
    assert isinstance(results, dict)
    # Add more assertions here based on the expected behavior of compute_metrics_over_data


def test_radar_chart(llm_monitoring_app):
    # Assuming radar_chart creates a chart and returns some form of result or output
    metrics_names = ["metric1", "metric2"]
    metrics_values = [0.5, 0.75]
    chart = llm_monitoring_app.radar_chart(metrics_names, metrics_values)
    # Add assertions here based on the expected behavior of radar_chart
