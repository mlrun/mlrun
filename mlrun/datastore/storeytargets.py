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
from urllib.parse import urlparse

import storey
from mergedeep import merge
from storey import V3ioDriver

import mlrun
from mlrun.datastore.base import DataStore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaStream,
    DatastoreProfileKafkaTarget,
    DatastoreProfilePostgreSQL,
    DatastoreProfileTDEngine,
    datastore_profile_read,
)

from ..platforms.iguazio import parse_path
from .utils import (
    parse_kafka_url,
)

"""
Storey targets expect storage_options, which may contain credentials.
To avoid passing it openly within the graph, we use wrapper classes.
"""


def get_url_and_storage_options(path, external_storage_options=None):
    store, resolved_store_path, url = mlrun.store_manager.get_or_create_store(path)
    storage_options = store.get_storage_options()
    if storage_options and external_storage_options:
        # merge external storage options with the store's storage options. storage_options takes precedence
        storage_options = merge(external_storage_options, storage_options)
    else:
        storage_options = storage_options or external_storage_options
    return url, DataStore._sanitize_options(storage_options)


class TDEngineStoreyTarget(storey.TDEngineTarget):
    def __init__(self, *args, url: str, **kwargs):
        if url.startswith("ds://"):
            datastore_profile = datastore_profile_read(url)
            if not isinstance(datastore_profile, DatastoreProfileTDEngine):
                raise ValueError(
                    f"Unexpected datastore profile type:{datastore_profile.type}."
                    "Only DatastoreProfileTDEngine is supported"
                )
            url = datastore_profile.dsn()
        kwargs["url"] = url
        super().__init__(*args, **kwargs)


class TimescaleDBStoreyTarget(storey.TimescaleDBTarget):
    def __init__(self, *args, url: str, **kwargs):
        if url.startswith("ds://"):
            datastore_profile = datastore_profile_read(url)
            if not isinstance(datastore_profile, DatastoreProfilePostgreSQL):
                raise ValueError(
                    f"Unexpected datastore profile type: {datastore_profile.type}. "
                    "Only DatastoreProfilePostgreSQL is supported"
                )
            url = datastore_profile.dsn()

        self._schema = None  # Remove - overridfing TimescaleDBTarget
        super().__init__(*args, dsn=url, **kwargs)

        # Remove - overridfing TimescaleDBTarget
        if self._schema is None and "." in self._table:
            self._schema, self._table = self._table.split(".", 1)

    # Remove - overridfing TimescaleDBTarget
    async def _emit(
        self, batch, batch_key, batch_time, batch_events, last_event_time=None
    ):
        """Write a batch of events to TimescaleDB.

        This method performs the core data writing functionality:
        1. Ensures the connection pool is initialized
        2. Converts dictionary events to tuples for efficient COPY operations
        3. Uses PostgreSQL's COPY protocol for high-performance bulk inserts
        4. Maintains proper column ordering for TimescaleDB compatibility

        Args:
            batch: list of events to write
            batch_key: Key used for batching (unused in this implementation)
            batch_time: Timestamp when batch was created
            batch_events: list of original event objects
            last_event_time: Timestamp of the most recent event in the batch
        """
        # Ensure connection pool is created
        await self._async_init()

        # Skip processing if batch is empty
        if not batch:
            return

        # Convert dictionaries to tuples for copy_records_to_table
        # PostgreSQL's COPY protocol requires data in tuple format with consistent column ordering

        records = []
        for item in batch:
            if not isinstance(item, dict):
                # Only dictionaries are supported as input
                raise TypeError(
                    f"TimescaleDBTarget only supports dictionary data, got {type(item)}"
                )

            # Convert dict to tuple in correct column order
            # This ensures time column is first, followed by data columns
            record = tuple(item.get(col) for col in self._column_names)
            records.append(record)
        # Write data using connection pool
        async with self._pool.acquire() as conn:
            # Use PostgreSQL's COPY protocol for optimal performance
            # This is significantly faster than individual INSERT statements
            await conn.copy_records_to_table(
                table_name=self._table,
                schema_name=self._schema,
                records=records,
                columns=self._column_names,
            )


class StoreyTargetUtils:
    @staticmethod
    def process_args_and_kwargs(args, kwargs):
        args = list(args)
        path = args[0] if args else kwargs.get("path")
        external_storage_options = kwargs.get("storage_options")

        url, storage_options = get_url_and_storage_options(
            path, external_storage_options
        )

        if storage_options:
            kwargs["storage_options"] = storage_options
        if args:
            args[0] = url
        if "path" in kwargs:
            kwargs["path"] = url
        return args, kwargs


class ParquetStoreyTarget(storey.ParquetTarget):
    def __init__(self, *args, **kwargs):
        alt_key_name = kwargs.pop("alternative_v3io_access_key", None)
        args, kwargs = StoreyTargetUtils.process_args_and_kwargs(args, kwargs)
        storage_options = kwargs.get("storage_options", {})
        if storage_options and storage_options.get("v3io_access_key") and alt_key_name:
            if alt_key := mlrun.get_secret_or_env(alt_key_name):
                storage_options["v3io_access_key"] = alt_key
        super().__init__(*args, **kwargs)


class CSVStoreyTarget(storey.CSVTarget):
    def __init__(self, *args, **kwargs):
        args, kwargs = StoreyTargetUtils.process_args_and_kwargs(args, kwargs)
        super().__init__(*args, **kwargs)


class StreamStoreyTarget(storey.StreamTarget):
    def __init__(self, *args, **kwargs):
        args = list(args)

        uri = args[0] if args else kwargs.get("stream_path")

        if not uri:
            raise mlrun.errors.MLRunInvalidArgumentError("StreamTarget requires a path")

        _, storage_options = get_url_and_storage_options(uri)
        _, path = parse_path(uri)

        access_key = storage_options.get("v3io_access_key")

        if alt_key_name := kwargs.pop("alternative_v3io_access_key", None):
            if alt_key := mlrun.get_secret_or_env(alt_key_name):
                access_key = alt_key

        storage = V3ioDriver(access_key=access_key)

        if storage_options:
            kwargs["storage"] = storage
        if args:
            args[0] = path
        if "stream_path" in kwargs:
            kwargs["stream_path"] = path

        super().__init__(*args, **kwargs)


class KafkaStoreyTarget(storey.KafkaTarget):
    def __init__(self, *args, **kwargs):
        kwargs.pop("alternative_v3io_access_key", None)
        path = kwargs.pop("path")
        attributes = kwargs.pop("attributes", {})
        if path and path.startswith("ds://"):
            datastore_profile = datastore_profile_read(path)
            if not isinstance(
                datastore_profile,
                (DatastoreProfileKafkaStream, DatastoreProfileKafkaTarget),
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Unsupported datastore profile type: {type(datastore_profile)}"
                )

            attributes = merge(attributes, datastore_profile.attributes())
            brokers = attributes.pop("brokers", None)
            # Override the topic with the one in the url (if any)
            parsed = urlparse(path)
            topic = (
                parsed.path.strip("/") if parsed.path else datastore_profile.get_topic()
            )
        else:
            brokers = attributes.pop("brokers", None)
            topic, brokers = parse_kafka_url(path, brokers)

        if not topic:
            raise mlrun.errors.MLRunInvalidArgumentError("KafkaTarget requires a topic")
        kwargs["brokers"] = brokers
        kwargs["topic"] = topic

        attributes = mlrun.datastore.utils.KafkaParameters(attributes).producer()

        super().__init__(*args, **kwargs, producer_options=attributes)


class NoSqlStoreyTarget(storey.NoSqlTarget):
    pass


class RedisNoSqlStoreyTarget(storey.NoSqlTarget):
    def __init__(self, *args, **kwargs):
        path = kwargs.pop("path")
        endpoint, uri = mlrun.datastore.targets.RedisNoSqlTarget.get_server_endpoint(
            path
        )
        kwargs["path"] = endpoint + "/" + uri
        super().__init__(*args, **kwargs)


class TSDBStoreyTarget(storey.TSDBTarget):
    pass
