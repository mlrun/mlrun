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

from uuid import UUID

from mlrun.model_monitoring.applications.evidently import _HAS_EVIDENTLY

if _HAS_EVIDENTLY:
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import DashboardPanelPlot
    from evidently.ui.workspace import (
        STR_UUID,
        OrgID,
        ProjectModel,
        Workspace,
        WorkspaceBase,
    )

_PROJECT_NAME = "Iris Monitoring"
_PROJECT_DESCRIPTION = "Test project using iris dataset"


def create_evidently_project(
    workspace: WorkspaceBase,
    id: UUID | None = None,
    org_id: OrgID | None = None,
):
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


def get_local_workspace(evidently_workspace_path: str) -> "Workspace":
    return Workspace.create(evidently_workspace_path)


def setup_evidently_project(
    evidently_project_id: "STR_UUID",
    evidently_workspace_path: str,
    org_id: OrgID | None = None,
) -> None:
    if isinstance(evidently_project_id, str):
        evidently_project_id = UUID(evidently_project_id)
    workspace = get_local_workspace(evidently_workspace_path)
    create_evidently_project(workspace, evidently_project_id, org_id=org_id)
