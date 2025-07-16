# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas
from mlrun.model_monitoring.db.tsdb.base import TSDBConnector


class TestTSDBConnectorStaticMethods:
    @pytest.fixture
    def results_data(self):
        """Fixture to create shared test data."""
        df = pd.DataFrame(
            {
                "result_kind": [0.0, 0.0, 0.0, 0.0],
                "application_name": ["my_app", "my_app", "my_app", "my_app"],
                "endpoint_id": ["mep_uid1", "mep_uid1", "mep_uid2", "mep_uid2"],
                "result_name": ["result1", "result2", "result1", "result3"],
            }
        )
        df["application_name"] = df["application_name"].astype("category")
        df["endpoint_id"] = df["endpoint_id"].astype("category")
        df["result_name"] = df["result_name"].astype("category")
        return df

    @pytest.fixture
    def metrics_data(self):
        """Fixture to create shared test data."""
        return pd.DataFrame(
            {
                "application_name": ["my_app", "my_app", "my_app", "my_app"],
                "endpoint_id": ["mep_uid1", "mep_uid1", "mep_uid2", "mep_uid2"],
                "metric_name": ["metric1", "metric2", "metric1", "metric3"],
            }
        )

    @pytest.fixture
    def empty_df(self):
        """Fixture to create shared test data."""
        return pd.DataFrame()

    def test_df_to_metrics_grouped_dict(self, results_data, metrics_data, empty_df):
        results_by_endpoint = TSDBConnector.df_to_metrics_grouped_dict(
            df=results_data, type="result", project="my_project"
        )
        assert ["result1", "result2"] == sorted(
            [result.name for result in results_by_endpoint["mep_uid1"]]
        )
        assert ["result1", "result3"] == sorted(
            [result.name for result in results_by_endpoint["mep_uid2"]]
        )

        metrics_by_endpoint = TSDBConnector.df_to_metrics_grouped_dict(
            df=metrics_data, type="metric", project="my_project"
        )
        assert ["metric1", "metric2"] == sorted(
            [metric.name for metric in metrics_by_endpoint["mep_uid1"]]
        )
        assert ["metric1", "metric3"] == sorted(
            [metric.name for metric in metrics_by_endpoint["mep_uid2"]]
        )

        # empty df test:
        assert (
            TSDBConnector.df_to_metrics_grouped_dict(
                df=empty_df, type="result", project="my_project"
            )
            == {}
        )
        assert (
            TSDBConnector.df_to_metrics_grouped_dict(
                df=empty_df, type="metric", project="my_project"
            )
            == {}
        )

    def test_df_to_metrics_list(self, results_data, metrics_data, empty_df):
        results = TSDBConnector.df_to_metrics_list(
            df=results_data, type="result", project="my_project"
        )
        assert ["result1", "result1", "result2", "result3"] == sorted(
            [result.name for result in results]
        )

        metrics = TSDBConnector.df_to_metrics_list(
            df=metrics_data, type="metric", project="my_project"
        )
        assert ["metric1", "metric1", "metric2", "metric3"] == sorted(
            [metric.name for metric in metrics]
        )

        # empty df test:
        assert (
            TSDBConnector.df_to_metrics_list(
                df=empty_df, type="result", project="my_project"
            )
            == []
        )
        assert (
            TSDBConnector.df_to_metrics_list(
                df=empty_df, type="metric", project="my_project"
            )
            == []
        )

    def test_df_to_events_intersection_dict(self, results_data, metrics_data, empty_df):
        metrics_key = mm_schemas.INTERSECT_DICT_KEYS[
            mm_schemas.ModelEndpointMonitoringMetricType.METRIC
        ]
        results_key = mm_schemas.INTERSECT_DICT_KEYS[
            mm_schemas.ModelEndpointMonitoringMetricType.RESULT
        ]
        result_intersection_dict = TSDBConnector.df_to_events_intersection_dict(
            df=results_data, type="result", project="my_project"
        )
        results = result_intersection_dict[results_key]
        assert len(results) == 1
        assert results[0].full_name == "my_project.my_app.result.result1"

        metric_intersection_dict = TSDBConnector.df_to_events_intersection_dict(
            df=metrics_data, type="metric", project="my_project"
        )

        metrics = metric_intersection_dict[metrics_key]
        assert len(metrics) == 1
        assert metrics[0].full_name == "my_project.my_app.metric.metric1"

        # empty df test:
        assert TSDBConnector.df_to_events_intersection_dict(
            df=empty_df, type="metric", project="my_project"
        ) == {metrics_key: []}
        assert TSDBConnector.df_to_events_intersection_dict(
            df=empty_df, type="result", project="my_project"
        ) == {results_key: []}
