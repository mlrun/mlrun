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

import enum
import typing

import mlrun.common.schemas.secret
import mlrun.errors

from .base import TSDBtarget


class ObjectTSDBFactory(enum.Enum):
    """Enum class to handle the different TSDB target type values for storing real time metrics"""

    v3io_tsdb = "v3io-tsdb"

    def to_tsdb_target(self, project: str, **kwargs) -> TSDBtarget:
        """
        Return a TSDBtarget object based on the provided enum value.
        :param project:                    The name of the project.
        :return: `TSDBtarget` object.
        """

        if self == self.v3io_tsdb:
            from .v3io.v3io import V3IOTSDBtarget

            return V3IOTSDBtarget(project=project, **kwargs)

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"{value} is not a valid tsdb, please choose a valid value: %{valid_values}."
        )


def get_tsdb_target(project: str, **kwargs) -> TSDBtarget:
    """
    Getting the TSDB target type based on mlrun.config.model_endpoint_monitoring.tsdb_target_type.
    :param project:         The name of the project.
    :return: `TSDBtarget` object. The main goal of this object is to handle different operations on the
             TSDB target such as updating drift metrics or write application record result.
    """

    # Get store type value from ObjectTSDBFactory enum class
    tsdb_target_type = ObjectTSDBFactory(
        mlrun.mlconf.model_endpoint_monitoring.tsdb_target_type
    )

    # Convert into TSDB store target object
    return tsdb_target_type.to_tsdb_target(project=project, **kwargs)
