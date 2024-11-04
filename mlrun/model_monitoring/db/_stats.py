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

import mlrun.datastore.base
from mlrun.common.schemas.model_monitoring.constants import StatsKind
from mlrun.model_monitoring.helpers import (
    get_monitoring_current_stats_data,
    get_monitoring_drift_measures_data,
    get_monitoring_stats_directory_path,
)
from mlrun.utils import logger


class ModelMonitoringJsonFile(AbstractContextManager):
    INITIAL_CONTENT = {}
    ENCODING = "utf-8"

    """
    Initialize applications monitoring json file object.
    The JSON file stores a dictionary of registered application name as key and Unix timestamp as value.
    When working with the schedules data, use this class as a context manager to read and write the data.
    """

    def __init__(self, item: mlrun.datastore.base.DataItem, file_type: str):
        self._path = item.url
        self._item = item
        self._file_type = file_type
        self._fs = cast(fsspec.AbstractFileSystem, self._item.store.filesystem)

    def create(self, data: Optional[dict] = None) -> None:
        """Create a json file with initial content - an empty dictionary"""
        logger.debug(
            f"Creating model monitoring {self._file_type} file", path=self._item.url
        )
        if data:
            self._data = data
            self._item.put(json.dumps(self._data))
        else:
            self._item.put(json.dumps(self.INITIAL_CONTENT))

    def delete(self) -> None:
        """Delete json file if it exists"""
        if self._fs.exists(self._path):
            logger.debug(
                f"Deleting model monitoring {self._file_type} file", path=self._item.url
            )
            self._item.delete()
        else:
            logger.debug(
                f"Model monitoring {self._file_type} file does not exist, nothing to delete",
                path=self._item.url,
            )

    def _open(self) -> None:
        try:
            self._data = json.loads(self._item.get().decode(encoding=self.ENCODING))
        except Exception as exc:
            logger.debug(
                f"Error while trying to read data for json file {self._file_type}", exc
            )

    def _close(self) -> None:
        self._item.put(json.dumps(self._data))

    def __enter__(self) -> "ModelMonitoringJsonFile":
        self._open()
        return super().__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        self._close()


class ModelMonitoringCurrentStatsFile(ModelMonitoringJsonFile):
    def __init__(self, project: str, endpoint_id: str) -> None:
        """
        Initialize File object specific for current stats.
        :param project:         (str) Project name
        :param endpoint_id:     (str) Endpoint name
        """
        super().__init__(
            get_monitoring_current_stats_data(project, endpoint_id),
            StatsKind.CURRENT_STATS.value,
        )

    @classmethod
    def from_model_endpoint(
        cls, model_endpoint: mlrun.common.schemas.ModelEndpoint
    ) -> "ModelMonitoringCurrentStatsFile":
        return cls(
            project=model_endpoint.metadata.project,
            endpoint_id=model_endpoint.metadata.uid,
        )


class ModelMonitoringDriftMeasuresFile(ModelMonitoringJsonFile):
    def __init__(self, project: str, endpoint_id: str) -> None:
        """
        Initialize File object specific for drift measures.
        :param project:         (str) Project name
        :param endpoint_id:     (str) Endpoint name
        """
        super().__init__(
            get_monitoring_drift_measures_data(project, endpoint_id),
            StatsKind.DRIFT_MEASURES.value,
        )

    @classmethod
    def from_model_endpoint(
        cls, model_endpoint: mlrun.common.schemas.ModelEndpoint
    ) -> "ModelMonitoringDriftMeasuresFile":
        return cls(
            project=model_endpoint.metadata.project,
            endpoint_id=model_endpoint.metadata.uid,
        )


def delete_model_monitoring_stats_folder(project: str) -> None:
    """Delete the model monitoring schedules folder of the project"""
    folder = get_monitoring_stats_directory_path(project)
    fs = mlrun.datastore.store_manager.object(folder).store.filesystem
    if fs and fs.exists(folder):
        logger.debug("Deleting model monitoring stats folder", folder=folder)
        fs.rm(folder, recursive=True)
    elif fs is None:  # In-memory store
        raise mlrun.errors.MLRunValueError(
            "Cannot delete a folder without a file-system"
        )
