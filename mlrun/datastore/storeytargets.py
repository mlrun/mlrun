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
import mlrun.model_monitoring.helpers
from mlrun.datastore.base import DataStore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileKafkaSource,
    DatastoreProfileKafkaTarget,
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
    return url, DataStore._sanitize_storage_options(storage_options)


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
        super().__init__(*args, url=url, **kwargs)


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
                (DatastoreProfileKafkaSource, DatastoreProfileKafkaTarget),
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
            brokers = attributes.pop(
                "brokers", attributes.pop("bootstrap_servers", None)
            )
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
            path,
            kwargs.pop("credentials_prefix", None),
        )
        kwargs["path"] = endpoint + "/" + uri
        super().__init__(*args, **kwargs)


class TSDBStoreyTarget(storey.TSDBTarget):
    pass
