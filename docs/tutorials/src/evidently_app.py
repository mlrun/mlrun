from typing import Optional
from uuid import UUID

import pandas as pd
from sklearn.datasets import load_iris

import mlrun.model_monitoring.applications.context as mm_context
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import (
    _HAS_EVIDENTLY,
    EvidentlyModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
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
    from evidently.ui.base import Project
    from evidently.ui.dashboards import (
        CounterAgg,
        DashboardConfig,
        DashboardPanelCounter,
        DashboardPanelPlot,
        PanelValue,
        PlotType,
        ReportFilter,
    )
    from evidently.ui.type_aliases import STR_UUID
    from evidently.ui.workspace import Workspace

    _PROJECT_NAME = "Iris Monitoring"
    _PROJECT_DESCRIPTION = "Test project using iris dataset"

    def _create_evidently_project(
        workspace: Workspace, id: Optional[UUID] = None
    ) -> Project:
        if id:
            project = Project(
                name=_PROJECT_NAME,
                description=_PROJECT_DESCRIPTION,
                dashboard=DashboardConfig(name=_PROJECT_NAME, panels=[]),
                id=id,
            )  # pyright: ignore[reportGeneralTypeIssues]
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


class DemoEvidentlyMonitoringApp(EvidentlyModelMonitoringApplicationBase):
    NAME = "evidently-app-test"

    def __init__(
        self,
        evidently_workspace_path: str,
        evidently_project_id: "STR_UUID",
    ) -> None:
        super().__init__(evidently_workspace_path, evidently_project_id)
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

        data_drift_report = self.create_report(
            sample_df, monitoring_context.end_infer_time
        )
        self.evidently_workspace.add_report(
            self.evidently_project_id, data_drift_report
        )
        data_drift_test_suite = self.create_test_suite(
            sample_df, monitoring_context.end_infer_time
        )
        self.evidently_workspace.add_test_suite(
            self.evidently_project_id, data_drift_test_suite
        )

        self.log_evidently_object(
            monitoring_context, data_drift_report, "evidently_report"
        )
        self.log_evidently_object(
            monitoring_context, data_drift_test_suite, "evidently_suite"
        )

        window_start = monitoring_context.start_infer_time
        window_end = monitoring_context.end_infer_time

        # Note: the times for evidently are those of the next monitoring window.
        evidently_start = window_end
        evidently_end = window_end + (window_end - window_start)

        self.log_project_dashboard(
            monitoring_context,
            timestamp_start=evidently_start,
            timestamp_end=evidently_end,
        )

        monitoring_context.logger.info("Logged evidently objects")
        return ModelMonitoringApplicationResult(
            name="data_drift_test",
            value=0.5,
            kind=ResultKindApp.data_drift,
            status=ResultStatusApp.potential_detection,
        )

    def create_report(
        self, sample_df: pd.DataFrame, schedule_time: pd.Timestamp
    ) -> "Report":
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
