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
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, cast

import fsspec

import mlrun.common.schemas
import mlrun.model_monitoring.helpers
from mlrun.utils import logger


class ModelMonitoringSchedulesFile(AbstractContextManager):
    INITIAL_CONTENT = json.dumps({})
    ENCODING = "utf-8"

    def __init__(self, project: str, endpoint_id: str) -> None:
        """
        Initialize applications monitoring schedules file object.
        The JSON file stores a dictionary of registered application name as key and Unix timestamp as value.
        When working with the schedules data, use this class as a context manager to read and write the data.

        :param project:     The project name.
        :param endpoint_id: The endpoint ID.
        """
        # `self._item` is the persistent version of the monitoring schedules.
        self._item = mlrun.model_monitoring.helpers.get_monitoring_schedules_data(
            project=project, endpoint_id=endpoint_id
        )
        self._path = self._item.url
        self._fs = cast(fsspec.AbstractFileSystem, self._item.store.filesystem)
        # `self._schedules` is an in-memory copy of the DB for all the applications for
        # the same model endpoint.
        self._schedules: dict[str, int] = {}

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

    def _open(self) -> None:
        self._schedules = json.loads(self._item.get().decode(encoding=self.ENCODING))

    def _close(self) -> None:
        self._item.put(json.dumps(self._schedules))

    def __enter__(self) -> "ModelMonitoringSchedulesFile":
        self._open()
        return super().__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self._close()

    def get_application_time(self, application: str) -> Optional[int]:
        return self._schedules.get(application)

    def update_application_time(self, application: str, timestamp: int) -> None:
        self._schedules[application] = timestamp


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
