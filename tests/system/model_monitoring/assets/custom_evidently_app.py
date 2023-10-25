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

import pandas as pd
from sklearn.datasets import load_iris

from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import ModelMonitoringApplicationResult
from mlrun.model_monitoring.evidently_application import (
    _HAS_EVIDENTLY,
    EvidentlyModelMonitoringApplication,
)

if _HAS_EVIDENTLY:
    from evidently.metrics import (
        ColumnDriftMetric,
        ColumnSummaryMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    from evidently.report import Report
    from evidently.test_preset import DataDriftTestPreset
    from evidently.test_suite import TestSuite


class CustomEvidentlyMonitoringApp(EvidentlyModelMonitoringApplication):
    name = "evidently-app-test"

    def run_application(
        self,
        application_name: str,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        schedule_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> ModelMonitoringApplicationResult:
        iris = load_iris()
        self.train_set = pd.DataFrame(
            iris.data,
            columns=[
                "sepal_length_cm",
                "sepal_width_cm",
                "petal_length_cm",
                "petal_width_cm",
            ],
        )

        sample_df = sample_df[
            ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
        ]

        data_drift_report = self.create_report(sample_df, schedule_time)
        self.evidently_workspace.add_report(
            self.evidently_project_id, data_drift_report
        )
        data_drift_test_suite = self.create_test_suite(sample_df, schedule_time)
        self.evidently_workspace.add_test_suite(
            self.evidently_project_id, data_drift_test_suite
        )

        self.log_evidently_object(data_drift_report, f"report_{str(schedule_time)}")
        self.log_evidently_object(data_drift_test_suite, f"suite_{str(schedule_time)}")
        self.log_project_dashboard(None, schedule_time + datetime.timedelta(minutes=1))

        return ModelMonitoringApplicationResult(
            self.name,
            endpoint_id,
            schedule_time,
            result_name="data_drift_test",
            result_value=0.5,
            result_kind=ResultKindApp.data_drift,
            result_status=ResultStatusApp.detected,
        )

    def create_report(
        self, sample_df: pd.DataFrame, schedule_time: pd.Timestamp
    ) -> "Report":
        metrics = [
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
        for col_name in [
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
        ]:
            metrics.extend(
                [
                    ColumnDriftMetric(column_name=col_name, stattest="wasserstein"),
                    ColumnSummaryMetric(column_name=col_name),
                ]
            )

        data_drift_report = Report(
            metrics=metrics,
            timestamp=schedule_time,
        )

        data_drift_report.run(reference_data=self.train_set, current_data=sample_df)
        return data_drift_report

    def create_test_suite(
        self, sample_df: pd.DataFrame, schedule_time: pd.Timestamp
    ) -> "TestSuite":
        data_drift_test_suite = TestSuite(
            tests=[DataDriftTestPreset()],
            timestamp=schedule_time,
        )

        data_drift_test_suite.run(reference_data=self.train_set, current_data=sample_df)
        return data_drift_test_suite
