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

# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

import enum
import typing
import warnings

import mlrun.common.schemas.secret
import mlrun.errors

from .base import StoreBase


class ObjectStoreFactory(enum.Enum):
    """Enum class to handle the different store type values for saving model monitoring records."""

    v3io_nosql = "v3io-nosql"
    SQL = "sql"

    def to_object_store(
        self,
        project: str,
        access_key: str = None,
        secret_provider: typing.Callable = None,
    ) -> StoreBase:
        """
        Return a StoreBase object based on the provided enum value.

        :param project:                   The name of the project.
        :param access_key:                Access key with permission to the DB table. Note that if access key is None
                                          and the endpoint target is from type KV then the access key will be
                                          retrieved from the environment variable.
        :param secret_provider:           An optional secret provider to get the connection string secret.

        :return: `StoreBase` object.

        """

        if self == self.v3io_nosql:
            from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase

            # Get V3IO access key from env
            access_key = access_key or mlrun.mlconf.get_v3io_access_key()

            return KVStoreBase(project=project, access_key=access_key)

        # Assuming SQL store target if store type is not KV.
        # Update these lines once there are more than two store target types.

        from mlrun.model_monitoring.db.stores.sqldb.sql_store import SQLStoreBase

        return SQLStoreBase(
            project=project,
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
) -> StoreBase:
    # Leaving here for backwards compatibility
    warnings.warn(
        "The 'get_model_endpoint_store' function is deprecated and will be removed in 1.9.0. "
        "Please use `get_store_object` instead.",
        # TODO: remove in 1.9.0
        FutureWarning,
    )
    return get_store_object(
        project=project, access_key=access_key, secret_provider=secret_provider
    )


def get_store_object(
    project: str,
    access_key: str = None,
    secret_provider: typing.Callable = None,
) -> StoreBase:
    """
    Getting the DB target type based on mlrun.config.model_endpoint_monitoring.store_type.

    :param project:         The name of the project.
    :param access_key:      Access key with permission to the DB table.
    :param secret_provider: An optional secret provider to get the connection string secret.

    :return: `StoreBase` object. Using this object, the user can apply different operations on the
             model monitoring record such as write, update, get and delete a model endpoint.
    """

    # Get store type value from ObjectStoreFactory enum class
    store_type = ObjectStoreFactory(mlrun.mlconf.model_endpoint_monitoring.store_type)

    # Convert into store target object
    return store_type.to_object_store(
        project=project, access_key=access_key, secret_provider=secret_provider
    )
