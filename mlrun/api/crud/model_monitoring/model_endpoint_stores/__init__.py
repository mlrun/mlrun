# Copyright 2018 Iguazio
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

import mlrun

from .model_endpoint_store import _ModelEndpointStore


class ModelEndpointStoreType(enum.Enum):
    """Enum class to handle the different store type values for saving a model endpoint record."""

    kv = "kv"
    sql = "sql"

    def to_endpoint_target(
        self,
        project: str,
        access_key: str = None,
        connection_string: str = None,
    ) -> _ModelEndpointStore:
        """
        Return a ModelEndpointStore object based on the provided enum value.

        :param project:           The name of the project.
        :param access_key:        Access key with permission to the DB table. Note that if access key is None and the
                                  endpoint target is from type KV then the access key will be retrieved from the
                                  environment variable.
        :param connection_string: A valid connection string for SQL target. Contains several key-value pairs that
                                  required for the database connection.
                                  e.g. A root user with password 1234, tries to connect a schema called mlrun within a
                                  local MySQL DB instance: 'mysql+pymysql://root:1234@localhost:3306/mlrun'.

        :return: ModelEndpointStore object.

        """

        if self.value == ModelEndpointStoreType.kv.value:

            from .kv_model_endpoint_store import _ModelEndpointKVStore

            # Get V3IO access key from env
            access_key = (
                mlrun.mlconf.get_v3io_access_key() if access_key is None else access_key
            )

            return _ModelEndpointKVStore(project=project, access_key=access_key)

        # Assuming SQL store target if store type is not KV.
        # Update these lines once there are more than two store target types.
        sql_connection_string = (
            connection_string
            if connection_string is not None
            else mlrun.mlconf.model_endpoint_monitoring.connection_string
        )
        from .sql_model_endpoint_store import _ModelEndpointSQLStore

        return _ModelEndpointSQLStore(
            project=project, connection_string=sql_connection_string
        )

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidArgumentError(
            "%r is not a valid %s, please choose a valid value: %s."
            % (value, cls.__name__, valid_values)
        )


def get_model_endpoint_target(
    project: str, access_key: str = None
) -> _ModelEndpointStore:
    """
    Getting the DB target type based on mlrun.config.model_endpoint_monitoring.store_type.

    :param project:    The name of the project.
    :param access_key: Access key with permission to the DB table.

    :return: ModelEndpointStore object. Using this object, the user can apply different operations on the
             model endpoint record such as write, update, get and delete.
    """

    # Get store type value from ModelEndpointStoreType enum class
    model_endpoint_store_type = ModelEndpointStoreType(
        mlrun.mlconf.model_endpoint_monitoring.store_type
    )

    # Convert into model endpoint store target object
    return model_endpoint_store_type.to_endpoint_target(project, access_key)
