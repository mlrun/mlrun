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

from typing import Optional
from uuid import UUID

import pandas as pd
from sklearn.datasets import load_iris

import mlrun.model_monitoring.applications.context as mm_context
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import ModelMonitoringApplicationResult
from mlrun.model_monitoring.applications.evidently import (
    _HAS_EVIDENTLY,
    EvidentlyModelMonitoringApplicationBase,
)

if _HAS_EVIDENTLY:
    from evidently.core.report import Report, Snapshot
    from evidently.legacy.metrics import (
        ColumnDriftMetric,
        ColumnSummaryMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    from evidently.legacy.test_preset import DataDriftTestPreset
    from evidently.legacy.test_suite import TestSuite
    from evidently.legacy.ui.dashboards import (
        CounterAgg,
        DashboardPanelCounter,
        DashboardPanelPlot,
        PanelValue,
        PlotType,
        ReportFilter,
    )
    from evidently.ui.workspace import (
        STR_UUID,
        Project,
        ProjectModel,
        WorkspaceBase,
    )

    _PROJECT_NAME = "Iris Monitoring"
    _PROJECT_DESCRIPTION = "Test project using iris dataset"

    def _create_evidently_project(
        workspace: WorkspaceBase, id: Optional[UUID] = None
    ) -> Project:
        if id:
            project = ProjectModel(
                name=_PROJECT_NAME, description=_PROJECT_DESCRIPTION, id=id
            )
            project = workspace.add_project(project)
        else:
            project = workspace.create_project(_PROJECT_NAME)
        project.description = _PROJECT_DESCRIPTION
        project.dashboard.add_panel(
            DashboardPanelCounter(
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                agg=CounterAgg.NONE,
                title="Income Dataset (iris)",
            )  # pyright: ignore[reportGeneralTypeIssues]
        )
        project.dashboard.add_panel(
            DashboardPanelCounter(
                title="Model Calls",
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                value=PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                    legend="count",
                ),
                text="count",
                agg=CounterAgg.SUM,
                size=1,
            )  # pyright: ignore[reportGeneralTypeIssues]
        )
        project.dashboard.add_panel(
            DashboardPanelCounter(
                title="Share of Drifted Features",
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                value=PanelValue(
                    metric_id="DatasetDriftMetric",
                    field_path="share_of_drifted_columns",
                    legend="share",
                ),
                text="share",
                agg=CounterAgg.LAST,
                size=1,
            )  # pyright: ignore[reportGeneralTypeIssues]
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Dataset Quality",
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                values=[
                    PanelValue(
                        metric_id="DatasetDriftMetric",
                        field_path="share_of_drifted_columns",
                        legend="Drift Share",
                    ),
                    PanelValue(
                        metric_id="DatasetMissingValuesMetric",
                        field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                        legend="Missing Values Share",
                    ),
                ],
                plot_type=PlotType.LINE,
            )  # pyright: ignore[reportGeneralTypeIssues]
        )
        project.save()
        return project


class CustomEvidentlyMonitoringApp(EvidentlyModelMonitoringApplicationBase):
    NAME = "evidently-app-test"

    def __init__(
        self,
        evidently_project_id: "STR_UUID",
        evidently_workspace_path: Optional[str] = None,
        cloud_workspace: bool = False,
    ) -> None:
        super().__init__(
            evidently_project_id=evidently_project_id,
            evidently_workspace_path=evidently_workspace_path,
            cloud_workspace=cloud_workspace,
        )
        self._init_evidently_project()
        self.train_set = None

    def _init_iris_data(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> None:
        if self.train_set is None:
            iris = load_iris()
            self.columns = monitoring_context.feature_names
            self.train_set = pd.DataFrame(iris.data, columns=self.columns)

    def _init_evidently_project(self) -> None:
        if self.evidently_project is None:
            if isinstance(self.evidently_project_id, str):
                self.evidently_project_id = UUID(self.evidently_project_id)
            self.evidently_project = _create_evidently_project(
                self.evidently_workspace, self.evidently_project_id
            )

    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> ModelMonitoringApplicationResult:
        self._init_iris_data(monitoring_context)
        monitoring_context.logger.info("Running evidently app")

        sample_df = monitoring_context.sample_df[self.columns]

        data_drift_report = self.create_report_run(
            sample_df, monitoring_context.end_infer_time
        )
        self.evidently_workspace.add_run(self.evidently_project_id, data_drift_report)

        data_drift_test_suite = self.create_test_suite(
            sample_df, monitoring_context.end_infer_time
        )
        self.evidently_workspace.add_run(
            self.evidently_project_id, data_drift_test_suite
        )

        self.log_evidently_object(
            monitoring_context, data_drift_report, "evidently_report"
        )
        self.log_evidently_object(
            monitoring_context, data_drift_test_suite, "evidently_suite"
        )

        monitoring_context.logger.info("Logged evidently objects")
        return ModelMonitoringApplicationResult(
            name="data_drift_test",
            value=0.5,
            kind=ResultKindApp.data_drift,
            status=ResultStatusApp.potential_detection,
        )

    def create_report_run(
        self, sample_df: pd.DataFrame, schedule_time: pd.Timestamp
    ) -> "Snapshot":
        metrics = [
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
        for col_name in self.columns:
            metrics.extend(
                [
                    ColumnDriftMetric(column_name=col_name, stattest="wasserstein"),
                    ColumnSummaryMetric(column_name=col_name),
                ]
            )

        data_drift_report = Report(
            metrics=metrics,
            metadata={"timestamp": str(schedule_time)},
        )

        return data_drift_report.run(
            reference_data=self.train_set, current_data=sample_df
        )

    def create_test_suite(
        self, sample_df: pd.DataFrame, schedule_time: pd.Timestamp
    ) -> "Snapshot":
        data_drift_test_suite = TestSuite(
            tests=[DataDriftTestPreset()],
            timestamp=schedule_time,
        )

        return data_drift_test_suite.run(
            reference_data=self.train_set, current_data=sample_df
        )
