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
import abc
import json
from abc import abstractmethod
from datetime import datetime, timezone
from typing import cast

import botocore.exceptions
import fsspec

import mlrun.datastore.base
from mlrun.common.schemas.model_monitoring.constants import StatsKind
from mlrun.model_monitoring.helpers import (
    get_monitoring_current_stats_data,
    get_monitoring_drift_measures_data,
    get_monitoring_stats_directory_path,
)
from mlrun.utils import logger


class ModelMonitoringStatsFile(abc.ABC):
    """
    Abstract class
    Initialize applications monitoring stats file object.
    The JSON file stores a dictionary of registered application name as key and Unix timestamp as value.
    When working with the schedules data, use this class as a context manager to read and write the data.
    """

    def __init__(self, item: mlrun.datastore.base.DataItem, file_type: str):
        self._path = item.url
        self._item = item
        self._file_type = file_type
        self._fs = cast(fsspec.AbstractFileSystem, self._item.store.filesystem)

    def create(self) -> None:
        """Create a json file with initial content - an empty dictionary"""
        logger.debug(
            f"Creating model monitoring {self._file_type} file", path=self._item.url
        )
        self._item.put(
            json.dumps(
                {
                    "data": dict(),
                    "timestamp": mlrun.utils.datetime_now().isoformat(
                        sep=" ", timespec="microseconds"
                    ),
                }
            )
        )

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

    def read(self) -> tuple[dict, datetime]:
        """
        Read the stats data and timestamp saved in file
        :return: tuple[dict, str] dictionary with stats data and timestamp saved in file
        """
        try:
            content = json.loads(self._item.get().decode())
            timestamp = content.get("timestamp")
            if timestamp is not None:
                timestamp = datetime.fromisoformat(timestamp).astimezone(
                    tz=timezone.utc
                )
            return content.get("data"), timestamp
        except (
            mlrun.errors.MLRunNotFoundError,
            # Different errors are raised for S3 or local storage, see ML-8042
            botocore.exceptions.ClientError,
            FileNotFoundError,
        ) as err:
            if (
                isinstance(err, botocore.exceptions.ClientError)
                # Add a log only to "NoSuchKey" errors codes - equivalent to `FileNotFoundError`
                and err.response["Error"]["Code"] != "NoSuchKey"
            ):
                raise

            logger.exception(
                "The Stats file was not found. It should have been created "
                "as a part of the model endpoint's creation",
                path=self._path,
                error=err,
            )
            raise

    def write(self, stats: dict, timestamp: datetime) -> None:
        """
        Write stats data to file overwrite the existing file
        :param stats: dictionary with the stats data
        :param timestamp: datetime object with the timestamp of last entry point for the stats calculation
        """
        content = {
            "data": stats,
            "timestamp": timestamp.isoformat(sep=" ", timespec="microseconds"),
        }
        self._item.put(json.dumps(content))

    @classmethod
    @abstractmethod
    def from_model_endpoint(
        cls, model_endpoint: mlrun.common.schemas.ModelEndpoint
    ) -> "ModelMonitoringStatsFile":
        """
        Return ModelMonitoringStatsFile child object using ModelEndpoint metadata
        :param model_endpoint: The current model endpoint to get a stats object for
        :return: ModelMonitoringStatsFile child object instance
        """
        pass


class ModelMonitoringCurrentStatsFile(ModelMonitoringStatsFile):
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


class ModelMonitoringDriftMeasuresFile(ModelMonitoringStatsFile):
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
