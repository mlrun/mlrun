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

import warnings
from abc import ABC
from tempfile import NamedTemporaryFile
from typing import Optional

import semver

import mlrun.model_monitoring.applications.base as mm_base
import mlrun.model_monitoring.applications.context as mm_context
from mlrun.errors import MLRunIncompatibleVersionError, MLRunValueError

SUPPORTED_EVIDENTLY_VERSION = semver.Version.parse("0.7.5")


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
    from evidently.core.report import Snapshot
    from evidently.ui.workspace import (
        STR_UUID,
        CloudWorkspace,
        Project,
        Workspace,
        WorkspaceBase,
    )


class EvidentlyModelMonitoringApplicationBase(
    mm_base.ModelMonitoringApplicationBase, ABC
):
    def __init__(
        self,
        evidently_project_id: "STR_UUID",
        evidently_workspace_path: Optional[str] = None,
        cloud_workspace: bool = False,
    ) -> None:
        """
        A class for integrating Evidently for MLRun model monitoring within a monitoring application.

        .. note::

            The ``evidently`` package is not installed by default in the mlrun/mlrun image.
            It must be installed separately to use this class.

        :param evidently_project_id:        (str) The ID of the Evidently project.
        :param evidently_workspace_path:    (str) The path to the Evidently workspace.
        :param cloud_workspace:             (bool) Whether the workspace is an Evidently Cloud workspace.
        """
        if not _HAS_EVIDENTLY:
            raise ModuleNotFoundError("Evidently is not installed - the app cannot run")
        self.evidently_workspace_path = evidently_workspace_path
        if cloud_workspace:
            self.get_workspace = self.get_cloud_workspace
        self.evidently_workspace = self.get_workspace()
        self.evidently_project_id = evidently_project_id
        self.evidently_project = self.load_project()

    def load_project(self) -> "Project":
        """Load the Evidently project."""
        return self.evidently_workspace.get_project(self.evidently_project_id)

    def get_workspace(self) -> "WorkspaceBase":
        """Get the Evidently workspace. Override this method for customize access to the workspace."""
        if self.evidently_workspace_path:
            return Workspace.create(self.evidently_workspace_path)
        else:
            raise MLRunValueError(
                "A local workspace could not be created as `evidently_workspace_path` is not set.\n"
                "If you intend to use a cloud workspace, please use `cloud_workspace=True` and set the "
                "`EVIDENTLY_API_KEY` environment variable. In other cases, override this method."
            )

    def get_cloud_workspace(self) -> "CloudWorkspace":
        """Load the Evidently cloud workspace according to the `EVIDENTLY_API_KEY` environment variable."""
        return CloudWorkspace()

    @staticmethod
    def log_evidently_object(
        monitoring_context: mm_context.MonitoringApplicationContext,
        evidently_object: "Snapshot",
        artifact_name: str,
        unique_per_endpoint: bool = True,
    ) -> None:
        """
        Logs an Evidently report or suite as an artifact.

        .. caution::

            Logging Evidently objects in every model monitoring window may cause scale issues.
            This method should be called on special occasions only.

        :param monitoring_context:  (MonitoringApplicationContext) The monitoring context to process.
        :param evidently_object:    (Snapshot) The Evidently run to log, e.g. a report run.
        :param artifact_name:       (str) The name for the logged artifact.
        :param unique_per_endpoint: by default ``True``, we will log different artifact for each model endpoint,
                                    set to ``False`` without changing item key will cause artifact override.
        """
        with NamedTemporaryFile(suffix=".html") as file:
            evidently_object.save_html(filename=file.name)
            monitoring_context.log_artifact(
                artifact_name,
                local_path=file.name,
                unique_per_endpoint=unique_per_endpoint,
            )
