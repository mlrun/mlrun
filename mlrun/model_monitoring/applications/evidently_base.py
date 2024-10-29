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

import uuid
import warnings
from abc import ABC

import pandas as pd
import semver

import mlrun.model_monitoring.applications.base as mm_base
import mlrun.model_monitoring.applications.context as mm_context
from mlrun.errors import MLRunIncompatibleVersionError

SUPPORTED_EVIDENTLY_VERSION = semver.Version.parse("0.4.32")


def _check_evidently_version(*, cur: semver.Version, ref: semver.Version) -> None:
    if ref.is_compatible(cur) or (
        cur.major == ref.major == 0 and cur.minor == ref.minor and cur.patch > ref.patch
    ):
        return
    if cur.major == ref.major == 0 and cur.minor > ref.minor:
        warnings.warn(
            f"Evidently version {cur} is not compatible with the tested "
            f"version {ref}, use at your own risk."
        )
    else:
        raise MLRunIncompatibleVersionError(
            f"Evidently version {cur} is not supported, please change to "
            f"{ref} (or another compatible version)."
        )


_HAS_EVIDENTLY = False
try:
    import evidently  # noqa: F401

    _check_evidently_version(
        cur=semver.Version.parse(evidently.__version__),
        ref=SUPPORTED_EVIDENTLY_VERSION,
    )
    _HAS_EVIDENTLY = True
except ModuleNotFoundError:
    pass


if _HAS_EVIDENTLY:
    from evidently.suite.base_suite import Display
    from evidently.ui.type_aliases import STR_UUID
    from evidently.ui.workspace import Workspace
    from evidently.utils.dashboard import TemplateParams, file_html_template


class EvidentlyModelMonitoringApplicationBase(
    mm_base.ModelMonitoringApplicationBase, ABC
):
    def __init__(
        self, evidently_workspace_path: str, evidently_project_id: "STR_UUID"
    ) -> None:
        """
        A class for integrating Evidently for mlrun model monitoring within a monitoring application.
        Note: evidently is not installed by default in the mlrun/mlrun image.
        It must be installed separately to use this class.

        :param evidently_workspace_path:    (str) The path to the Evidently workspace.
        :param evidently_project_id:        (str) The ID of the Evidently project.

        """

        # TODO : more then one project (mep -> project)
        if not _HAS_EVIDENTLY:
            raise ModuleNotFoundError("Evidently is not installed - the app cannot run")
        self.evidently_workspace = Workspace.create(evidently_workspace_path)
        self.evidently_project_id = evidently_project_id
        self.evidently_project = self.evidently_workspace.get_project(
            evidently_project_id
        )

    @staticmethod
    def log_evidently_object(
        monitoring_context: mm_context.MonitoringApplicationContext,
        evidently_object: "Display",
        artifact_name: str,
    ) -> None:
        """
         Logs an Evidently report or suite as an artifact.

        :param monitoring_context:  (MonitoringApplicationContext) The monitoring context to process.
        :param evidently_object:    (Display) The Evidently display to log, e.g. a report or a test suite object.
        :param artifact_name:       (str) The name for the logged artifact.
        """
        evidently_object_html = evidently_object.get_html()
        monitoring_context.log_artifact(
            artifact_name, body=evidently_object_html.encode("utf-8"), format="html"
        )

    def log_project_dashboard(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
        timestamp_start: pd.Timestamp,
        timestamp_end: pd.Timestamp,
        artifact_name: str = "dashboard",
    ) -> None:
        """
        Logs an Evidently project dashboard.

        :param monitoring_context:  (MonitoringApplicationContext) The monitoring context to process.
        :param timestamp_start:     (pd.Timestamp) The start timestamp for the dashboard data.
        :param timestamp_end:       (pd.Timestamp) The end timestamp for the dashboard data.
        :param artifact_name:       (str) The name for the logged artifact.
        """

        dashboard_info = self.evidently_project.build_dashboard_info(
            timestamp_start, timestamp_end
        )
        template_params = TemplateParams(
            dashboard_id="pd_" + str(uuid.uuid4()).replace("-", ""),
            dashboard_info=dashboard_info,
            additional_graphs={},
        )

        dashboard_html = file_html_template(params=template_params)
        monitoring_context.log_artifact(
            artifact_name, body=dashboard_html.encode("utf-8"), format="html"
        )
