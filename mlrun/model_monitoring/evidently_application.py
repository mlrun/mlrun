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
#
import uuid
from typing import Union

import pandas as pd
from evidently.renderers.notebook_utils import determine_template
from evidently.report.report import Report
from evidently.suite.base_suite import Suite
from evidently.ui.workspace import Workspace
from evidently.utils.dashboard import TemplateParams

from mlrun.model_monitoring.application import ModelMonitoringApplication


class EvidentlyModelMonitoringApplication(ModelMonitoringApplication):
    def __init__(
        self, evidently_workspace_path: str = None, evidently_project_id: str = None
    ):
        """
        A class for integrating Evidently for mlrun model monitoring within a monitoring application.

        :param evidently_workspace_path:    (str) The path to the Evidently workspace.
        :param evidently_project_id:        (str) The ID of the Evidently project.

        """
        self.evidently_workspace = Workspace.create(evidently_workspace_path)
        self.evidently_project_id = evidently_project_id
        self.evidently_project = self.evidently_workspace.get_project(
            evidently_project_id
        )

    def log_evidently_object(
        self, evidently_object: Union[Report, Suite], artifact_name: str
    ):
        """
         Logs an Evidently report or suite as an artifact.

        :param evidently_object:    (Union[Report, Suite]) The Evidently report or suite object.
        :param artifact_name:       (str) The name for the logged artifact.
        """
        evidently_object_html = evidently_object.get_html()
        self.context.log_artifact(
            artifact_name, body=evidently_object_html.encode("utf-8"), format="html"
        )

    def log_project_dashboard(
        self,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        artifact_name: str = "dashboard",
    ):
        """
        Logs an Evidently project dashboard.

        :param timestamp_start: (pd.Timestamp) The start timestamp for the dashboard data.
        :param timestamp_end:   (pd.Timestamp) The end timestamp for the dashboard data.
        :param artifact_name:   (str) The name for the logged artifact.
        """

        dashboard_info = self.evidently_project.build_dashboard_info(
            timestamp_start, timestamp_end
        )
        template_params = TemplateParams(
            dashboard_id="pd_" + str(uuid.uuid4()).replace("-", ""),
            dashboard_info=dashboard_info,
            additional_graphs={},
        )

        dashboard_html = self._render(determine_template("inline"), template_params)
        self.context.log_artifact(
            artifact_name, body=dashboard_html.encode("utf-8"), format="html"
        )

    @staticmethod
    def _render(temple_func, template_params: TemplateParams):
        return temple_func(params=template_params)
