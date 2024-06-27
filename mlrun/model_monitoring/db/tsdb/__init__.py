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

from .base import TSDBConnector


class ObjectTSDBFactory(enum.Enum):
    """Enum class to handle the different TSDB connector type values for storing real time metrics"""

    v3io_tsdb = "v3io-tsdb"
    tdengine = "tdengine"

    def to_tsdb_connector(self, project: str, **kwargs) -> TSDBConnector:
        """
        Return a TSDBConnector object based on the provided enum value.
        :param project: The name of the project.
        :return: `TSDBConnector` object.
        """

        if self == self.v3io_tsdb:
            if mlrun.mlconf.is_ce_mode():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"{self.v3io_tsdb} is not supported in CE mode."
                )

            from .v3io.v3io_connector import V3IOTSDBConnector

            return V3IOTSDBConnector(project=project, **kwargs)

        # Assuming TDEngine connector if connector type is not V3IO TSDB.
        # Update these lines once there are more than two connector types.

        from .tdengine.tdengine_connector import TDEngineConnector

        return TDEngineConnector(project=project, **kwargs)

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"{value} is not a valid tsdb, please choose a valid value: %{valid_values}."
        )


def get_tsdb_connector(
    project: str,
    tsdb_connector_type: str = "",
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
    **kwargs,
) -> TSDBConnector:
    """
    Get TSDB connector object.
    :param project: The name of the project.
    :param tsdb_connector_type: The type of the TSDB connector. See mlrun.model_monitoring.db.tsdb.ObjectTSDBFactory
                                for available options.
    :param secret_provider: An optional secret provider to get the connection string secret.

    :return: `TSDBConnector` object. The main goal of this object is to handle different operations on the
             TSDB connector such as updating drift metrics or write application record result.
    """

    tsdb_connection_string = mlrun.model_monitoring.helpers.get_tsdb_connection_string(
        secret_provider=secret_provider
    )

    if tsdb_connection_string and tsdb_connection_string.startswith("taosws"):
        tsdb_connector_type = mlrun.common.schemas.model_monitoring.TSDBTarget.TDEngine
        kwargs["connection_string"] = tsdb_connection_string

    # Set the default TSDB connector type if no connection has been set
    tsdb_connector_type = (
        tsdb_connector_type
        or mlrun.mlconf.model_endpoint_monitoring.tsdb_connector_type
    )

    # Get connector type value from ObjectTSDBFactory enum class
    tsdb_connector_factory = ObjectTSDBFactory(tsdb_connector_type)

    # Convert into TSDB connector object
    return tsdb_connector_factory.to_tsdb_connector(project=project, **kwargs)
