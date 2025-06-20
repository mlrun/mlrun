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
import mlrun.datastore.datastore_profile
import mlrun.errors
import mlrun.model_monitoring.helpers
from mlrun.datastore.datastore_profile import DatastoreProfile

from .base import TSDBConnector


class ObjectTSDBFactory(enum.Enum):
    """Enum class to handle the different TSDB connector type values for storing real time metrics"""

    v3io_tsdb = "v3io-tsdb"
    tdengine = "tdengine"

    def to_tsdb_connector(
        self, project: str, profile: DatastoreProfile, **kwargs
    ) -> TSDBConnector:
        """
        Return a TSDBConnector object based on the provided enum value.
        :param project: The name of the project.
        :param profile: Datastore profile containing DSN and credentials for TSDB connection
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

        return TDEngineConnector(project=project, profile=profile, **kwargs)

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidMMStoreTypeError(
            f"{value} is not a valid tsdb, please choose a valid value: %{valid_values}."
        )


def get_tsdb_connector(
    project: str,
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
    profile: typing.Optional[mlrun.datastore.datastore_profile.DatastoreProfile] = None,
) -> TSDBConnector:
    """
    Get TSDB connector object.
    :param project:                 The name of the project.
    :param secret_provider:         An optional secret provider to get the connection string secret.
    :param profile:                 An optional profile to initialize the TSDB connector from.

    :return: ``TSDBConnector`` object. The main goal of this object is to handle different operations on the
             TSDB connector such as updating drift metrics or write application record result.
    :raise: ``MLRunNotFoundError`` if the user didn't set the TSDB datastore profile and didn't provide it through
            the ``profile`` parameter.
    :raise: ``MLRunInvalidMMStoreTypeError`` if the TSDB datastore profile is of an invalid type.
    """
    profile = profile or mlrun.model_monitoring.helpers._get_tsdb_profile(
        project=project, secret_provider=secret_provider
    )
    kwargs = {}
    if isinstance(profile, mlrun.datastore.datastore_profile.DatastoreProfileV3io):
        tsdb_connector_type = mlrun.common.schemas.model_monitoring.TSDBTarget.V3IO_TSDB
    elif isinstance(
        profile, mlrun.datastore.datastore_profile.DatastoreProfileTDEngine
    ):
        tsdb_connector_type = mlrun.common.schemas.model_monitoring.TSDBTarget.TDEngine
    else:
        extra_message = (
            ""
            if profile
            else " by using `project.set_model_monitoring_credentials` API"
        )
        raise mlrun.errors.MLRunInvalidMMStoreTypeError(
            "You must provide a valid TSDB datastore profile"
            f"{extra_message}. "
            f"Found an unexpected profile of class: {type(profile)}"
        )

    # Get connector type value from ObjectTSDBFactory enum class
    tsdb_connector_factory = ObjectTSDBFactory(tsdb_connector_type)

    # Convert into TSDB connector object
    return tsdb_connector_factory.to_tsdb_connector(
        project=project, profile=profile, **kwargs
    )
