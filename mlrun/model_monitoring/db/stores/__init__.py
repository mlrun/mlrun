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
        **kwargs,
    ) -> StoreBase:
        """
        Return a StoreBase object based on the provided enum value.

        :param project:                   The name of the project.

        :return: `StoreBase` object.

        """

        if self == self.v3io_nosql:
            from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase

            return KVStoreBase(project=project)

        # Assuming SQL store target if store type is not KV.
        # Update these lines once there are more than two store target types.

        from mlrun.model_monitoring.db.stores.sqldb.sql_store import SQLStoreBase

        return SQLStoreBase(
            project=project,
            **kwargs,
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
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
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
    secret_provider: typing.Optional[typing.Callable[[str], str]] = None,
    **kwargs,
) -> StoreBase:
    """
    Generate a store object. If a connection string is provided, the store type will be updated according to the
    connection string. Currently, the supported store types are SQL and v3io-nosql.

    :param project:         The name of the project.
    :param secret_provider: An optional secret provider to get the connection string secret.

    :return: `StoreBase` object. Using this object, the user can apply different operations such as write, update, get
    and delete a model endpoint record.
    """

    store_connection_string = mlrun.model_monitoring.helpers.get_connection_string(
        secret_provider=secret_provider
    )

    if store_connection_string and (
        store_connection_string.startswith("mysql")
        or store_connection_string.startswith("sqlite")
    ):
        store_type = mlrun.common.schemas.model_monitoring.ModelEndpointTarget.SQL
        kwargs["store_connection_string"] = store_connection_string
    elif store_connection_string and store_connection_string == "v3io":
        store_type = (
            mlrun.common.schemas.model_monitoring.ModelEndpointTarget.V3IO_NOSQL
        )
    else:
        store_type = None
    # Get store type value from ObjectStoreFactory enum class
    store_type_fact = ObjectStoreFactory(store_type)

    # Convert into store target object
    return store_type_fact.to_object_store(
        project=project, secret_provider=secret_provider, **kwargs
    )
