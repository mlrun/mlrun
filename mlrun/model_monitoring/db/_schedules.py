# Copyright 2024 Iguazio
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

import json
from typing import cast

import fsspec

import mlrun.common.schemas
import mlrun.model_monitoring.helpers
from mlrun.utils import logger


class ModelMonitoringSchedulesFile:
    INITIAL_CONTENT = json.dumps({})

    def __init__(self, project: str, endpoint_id: str) -> None:
        self._item = mlrun.model_monitoring.helpers.get_monitoring_schedules_data(
            project=project, endpoint_id=endpoint_id
        )
        self._path = self._item.url
        self._fs = cast(fsspec.AbstractFileSystem, self._item.store.filesystem)

    @classmethod
    def from_model_endpoint(
        cls, model_endpoint: mlrun.common.schemas.ModelEndpoint
    ) -> "ModelMonitoringSchedulesFile":
        return cls(
            project=model_endpoint.metadata.project,
            endpoint_id=model_endpoint.metadata.uid,
        )

    def create(self) -> None:
        """Create a schedules file with initial content - an empty dictionary"""
        logger.debug("Creating model monitoring schedules file", path=self._item.url)
        self._item.put(self.INITIAL_CONTENT)

    def delete(self) -> None:
        """Delete schedules file if it exists"""
        if self._fs.exists(self._path):
            logger.debug(
                "Deleting model monitoring schedules file", path=self._item.url
            )
            self._item.delete()
        else:
            logger.debug(
                "Model monitoring schedules file does not exist, nothing to delete",
                path=self._item.url,
            )


def delete_model_monitoring_schedules_folder(project: str) -> None:
    """Delete the model monitoring schedules folder of the project"""
    folder = mlrun.model_monitoring.helpers._get_monitoring_schedules_folder_path(
        project
    )
    fs = cast(
        fsspec.AbstractFileSystem,
        mlrun.datastore.store_manager.object(folder).store.filesystem,
    )
    if fs.exists(folder):
        logger.debug("Deleting model monitoring schedules folder", folder=folder)
        fs.rm(folder, recursive=True)
