# Copyright 2025 Iguazio
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
from mlrun.feature_store.api import norm_column_name
from mlrun.model_monitoring.applications import ModelMonitoringApplicationResult
from mlrun.model_monitoring.applications.evidently import (
    _HAS_EVIDENTLY,
    EvidentlyModelMonitoringApplicationBase,
)

if _HAS_EVIDENTLY:
    from evidently.core.report import Report, Snapshot
    from evidently.metrics import DatasetMissingValueCount, ValueDrift
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import DashboardPanelPlot
    from evidently.ui.workspace import (
        STR_UUID,
        OrgID,
        Project,
        ProjectModel,
        WorkspaceBase,
    )

    _PROJECT_NAME = "Iris Monitoring"
    _PROJECT_DESCRIPTION = "Test project using iris dataset"

    def _create_evidently_project(
        workspace: WorkspaceBase,
        id: Optional[UUID] = None,
        org_id: Optional[OrgID] = None,
    ) -> Project:
        if id:
            project = ProjectModel(
                name=_PROJECT_NAME, description=_PROJECT_DESCRIPTION, id=id
            )
            project = workspace.add_project(project, org_id=org_id)
        else:
            project = workspace.create_project(_PROJECT_NAME, org_id=org_id)
        project.description = _PROJECT_DESCRIPTION
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Income Dataset (iris)",
                subtitle="The iris dataset.",
                size="half",
                values=[PanelMetric(legend="Row count", metric="RowCount")],
                plot_params={"plot_type": "counter", "aggregation": "sum"},
            ),
            tab="tab 0",
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Model Calls",
                subtitle="Total number of predictions over time.",
                size="half",
                values=[PanelMetric(legend="count", metric="DatasetMissingValueCount")],
                plot_params={"plot_type": "counter", "aggregation": "sum"},
            ),
            tab="tab 0",
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Share of Drifted Features",
                subtitle="Measure the drift of the features.",
                size="full",
                values=[PanelMetric(metric="DataDriftPreset", legend="share")],
                plot_params={"plot_type": "counter", "aggregation": "last"},
            ),
            tab="tab 0",
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Dataset Quality",
                subtitle="",
                size="full",
                values=[
                    PanelMetric(
                        metric="DataDriftPreset",
                        legend="Drift Share",
                    ),
                    PanelMetric(
                        metric="DatasetMissingValuesMetric",
                        legend="Missing Values Share",
                    ),
                ],
                plot_params={"plot_type": "line"},
            ),
            tab="tab 0",
        )
        project.save()
        return project


class DemoEvidentlyMonitoringApp(EvidentlyModelMonitoringApplicationBase):
    NAME = "evidently-app-test"

    def __init__(
        self,
        evidently_project_id: Optional["STR_UUID"] = None,
        evidently_workspace_path: Optional[str] = None,
        cloud_workspace: bool = False,
        evidently_organization_id: Optional["OrgID"] = None,
    ) -> None:
        self.org_id = evidently_organization_id
        self._init_iris_data()
        super().__init__(
            evidently_project_id=evidently_project_id,
            evidently_workspace_path=evidently_workspace_path,
            cloud_workspace=cloud_workspace,
        )

    def _init_iris_data(self) -> None:
        iris = load_iris()
        self.columns = [norm_column_name(col) for col in iris.feature_names]
        self.train_set = pd.DataFrame(iris.data, columns=self.columns)

    def load_project(self) -> None:
        if isinstance(self.evidently_project_id, str):
            self.evidently_project_id = UUID(self.evidently_project_id)
        self.evidently_project = _create_evidently_project(
            self.evidently_workspace, self.evidently_project_id, org_id=self.org_id
        )
        self.evidently_project_id = self.evidently_project.id

    def do_tracking(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info("Running evidently app")

        sample_df = monitoring_context.sample_df[self.columns]

        data_drift_report_run = self.create_report_run(
            sample_df, monitoring_context.end_infer_time
        )
        self.evidently_workspace.add_run(
            self.evidently_project_id, data_drift_report_run
        )

        self.log_evidently_object(
            monitoring_context, data_drift_report_run, "evidently_report"
        )
        monitoring_context.logger.info("Logged evidently object")

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
            DataDriftPreset(),
            DatasetMissingValueCount(),
            DataSummaryPreset(),
        ]
        metrics.extend(
            [
                ValueDrift(column=col_name, method="wasserstein")
                for col_name in self.columns
            ]
        )

        data_drift_report = Report(
            metrics=metrics,
            metadata={"timestamp": str(schedule_time)},
            include_tests=True,
        )

        return data_drift_report.run(
            current_data=sample_df, reference_data=self.train_set
        )
