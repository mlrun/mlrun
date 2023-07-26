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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import enum
import typing

import mlrun.common.schemas.secret
import mlrun.errors

from .model_endpoint_store import ModelEndpointStore


class ModelEndpointStoreType(enum.Enum):
    """Enum class to handle the different store type values for saving a model endpoint record."""

    v3io_nosql = "v3io-nosql"
    SQL = "sql"

    def to_endpoint_store(
        self,
        project: str,
        access_key: str = None,
        endpoint_store_connection: str = None,
        secret_provider: typing.Callable = None,
    ) -> ModelEndpointStore:
        """
        Return a ModelEndpointStore object based on the provided enum value.

        :param project:                    The name of the project.
        :param access_key:                 Access key with permission to the DB table. Note that if access key is None
                                           and the endpoint target is from type KV then the access key will be
                                           retrieved from the environment variable.
        :param endpoint_store_connection: A valid connection string for model endpoint target. Contains several
                                          key-value pairs that required for the database connection.
                                          e.g. A root user with password 1234, tries to connect a schema called
                                          mlrun within a local MySQL DB instance:
                                          'mysql+pymysql://root:1234@localhost:3306/mlrun'.
        :param secret_provider:           An optional secret provider to get the connection string secret.

        :return: `ModelEndpointStore` object.

        """

        if self.value == ModelEndpointStoreType.v3io_nosql.value:
            from .kv_model_endpoint_store import KVModelEndpointStore

            # Get V3IO access key from env
            access_key = access_key or mlrun.mlconf.get_v3io_access_key()

            return KVModelEndpointStore(project=project, access_key=access_key)

        # Assuming SQL store target if store type is not KV.
        # Update these lines once there are more than two store target types.

        from .sql_model_endpoint_store import SQLModelEndpointStore

        return SQLModelEndpointStore(
            project=project,
            sql_connection_string=endpoint_store_connection,
            secret_provider=secret_provider,
        )

    @classmethod
    def _missing_(cls, value: typing.Any):
        """A lookup function to handle an invalid value.
        :param value: Provided enum (invalid) value.
        """
        valid_values = list(cls.__members__.keys())
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"{value} is not a valid endpoint store, please choose a valid value: %{valid_values}."
        )


def get_model_endpoint_store(
    project: str,
    access_key: str = None,
    secret_provider: typing.Callable = None,
) -> ModelEndpointStore:
    """
    Getting the DB target type based on mlrun.config.model_endpoint_monitoring.store_type.

    :param project:         The name of the project.
    :param access_key:      Access key with permission to the DB table.
    :param secret_provider: An optional secret provider to get the connection string secret.

    :return: `ModelEndpointStore` object. Using this object, the user can apply different operations on the
             model endpoint record such as write, update, get and delete.
    """

    # Get store type value from ModelEndpointStoreType enum class
    model_endpoint_store_type = ModelEndpointStoreType(
        mlrun.mlconf.model_endpoint_monitoring.store_type
    )

    # Convert into model endpoint store target object
    return model_endpoint_store_type.to_endpoint_store(
        project=project, access_key=access_key, secret_provider=secret_provider
    )
