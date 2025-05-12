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
#
import math
import tarfile
import tempfile
import typing
import warnings
from urllib.parse import parse_qs, urlparse

import pandas as pd
import semver

import mlrun.datastore


def parse_kafka_url(
    url: str, brokers: typing.Optional[typing.Union[list, str]] = None
) -> tuple[str, list]:
    """Generating Kafka topic and adjusting a list of bootstrap servers.

    :param url:               URL path to parse using urllib.parse.urlparse.
    :param brokers: List of kafka brokers.

    :return: A tuple of:
         [0] = Kafka topic value
         [1] = List of bootstrap servers
    """
    brokers = brokers or []

    if isinstance(brokers, str):
        brokers = brokers.split(",")

    # Parse the provided URL into six components according to the general structure of a URL
    url = urlparse(url)

    # Add the network location to the bootstrap servers list
    if url.netloc:
        brokers = [url.netloc] + brokers

    # Get the topic value from the parsed url
    query_dict = parse_qs(url.query)
    if "topic" in query_dict:
        topic = query_dict["topic"][0]
    else:
        topic = url.path
        topic = topic.lstrip("/")
    return topic, brokers


def upload_tarball(source_dir, target, secrets=None):
    # will delete the temp file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_fh:
        with tarfile.open(mode="w:gz", fileobj=temp_fh) as tar:
            tar.add(source_dir, arcname="")
        stores = mlrun.datastore.store_manager.set(secrets)
        datastore, subpath, url = stores.get_or_create_store(target)
        datastore.upload(subpath, temp_fh.name)


def filter_df_start_end_time(
    df: typing.Union[pd.DataFrame, typing.Iterator[pd.DataFrame]],
    time_column: typing.Optional[str] = None,
    start_time: pd.Timestamp = None,
    end_time: pd.Timestamp = None,
) -> typing.Union[pd.DataFrame, typing.Iterator[pd.DataFrame]]:
    if not time_column:
        return df
    if isinstance(df, pd.DataFrame):
        return _execute_time_filter(df, time_column, start_time, end_time)
    else:
        return filter_df_generator(df, time_column, start_time, end_time)


def filter_df_generator(
    dfs: typing.Iterator[pd.DataFrame],
    time_field: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> typing.Iterator[pd.DataFrame]:
    for df in dfs:
        yield _execute_time_filter(df, time_field, start_time, end_time)


def _execute_time_filter(
    df: pd.DataFrame, time_column: str, start_time: pd.Timestamp, end_time: pd.Timestamp
):
    if semver.parse(pd.__version__)["major"] >= 2:
        # pandas 2 is too strict by default (ML-5629)
        kwargs = {
            "format": "mixed",
            "yearfirst": True,
        }
    else:
        # pandas 1 may fail on format "mixed" (ML-5661)
        kwargs = {}
    df[time_column] = pd.to_datetime(df[time_column], **kwargs)
    if start_time:
        df = df[df[time_column] > start_time]
    if end_time:
        df = df[df[time_column] <= end_time]
    return df


def select_columns_from_df(
    df: typing.Union[pd.DataFrame, typing.Iterator[pd.DataFrame]],
    columns: list[str],
) -> typing.Union[pd.DataFrame, typing.Iterator[pd.DataFrame]]:
    if not columns:
        return df
    if isinstance(df, pd.DataFrame):
        return df[columns]
    else:
        return select_columns_generator(df, columns)


def select_columns_generator(
    dfs: typing.Union[pd.DataFrame, typing.Iterator[pd.DataFrame]],
    columns: list[str],
) -> typing.Iterator[pd.DataFrame]:
    for df in dfs:
        yield df[columns]


def _generate_sql_query_with_time_filter(
    table_name: str,
    engine: "sqlalchemy.engine.Engine",  # noqa: F821,
    time_column: str,
    parse_dates: list[str],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
):
    # Validate sqlalchemy (not installed by default):
    try:
        import sqlalchemy
    except (ModuleNotFoundError, ImportError) as exc:
        raise mlrun.errors.MLRunMissingDependencyError(
            "Using 'SQLTarget' requires sqlalchemy package. Use pip install mlrun[sqlalchemy] to install it."
        ) from exc
    table = sqlalchemy.Table(
        table_name,
        sqlalchemy.MetaData(),
        autoload=True,
        autoload_with=engine,
    )
    query = sqlalchemy.select(table)
    if time_column:
        if parse_dates and time_column not in parse_dates:
            parse_dates.append(time_column)
        else:
            parse_dates = [time_column]
        if start_time:
            query = query.filter(getattr(table.c, time_column) > start_time)
        if end_time:
            query = query.filter(getattr(table.c, time_column) <= end_time)

    return query, parse_dates


def get_kafka_brokers_from_dict(options: dict, pop=False) -> typing.Optional[str]:
    get_or_pop = options.pop if pop else options.get
    kafka_brokers = get_or_pop("kafka_brokers", None)
    if kafka_brokers:
        return kafka_brokers
    kafka_bootstrap_servers = get_or_pop("kafka_bootstrap_servers", None)
    if kafka_bootstrap_servers:
        warnings.warn(
            "The 'kafka_bootstrap_servers' parameter is deprecated in 1.7.0 and will be removed in "
            "1.10.0. Please pass the 'kafka_brokers' parameter instead.",
            FutureWarning,
        )
    return kafka_bootstrap_servers


def transform_list_filters_to_tuple(additional_filters):
    tuple_filters = []
    if not additional_filters:
        return tuple_filters
    validate_additional_filters(additional_filters)
    for additional_filter in additional_filters:
        tuple_filters.append(tuple(additional_filter))
    return tuple_filters


def validate_additional_filters(additional_filters):
    nan_error_message = "using NaN in additional_filters is not supported"
    if additional_filters in [None, [], ()]:
        return
    for filter_tuple in additional_filters:
        if filter_tuple == () or filter_tuple == []:
            continue
        if not isinstance(filter_tuple, (list, tuple)):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"mlrun supports additional_filters only as a list of tuples."
                f" Current additional_filters: {additional_filters}"
            )
        if isinstance(filter_tuple[0], (list, tuple)):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"additional_filters does not support nested list inside filter tuples except in -in- logic."
                f" Current filter_tuple: {filter_tuple}."
            )
        if len(filter_tuple) != 3:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"illegal filter tuple length, {filter_tuple} in additional filters:"
                f" {additional_filters}"
            )
        col_name, op, value = filter_tuple
        if isinstance(value, float) and math.isnan(value):
            raise mlrun.errors.MLRunInvalidArgumentError(nan_error_message)
        elif isinstance(value, (list, tuple)):
            for sub_value in value:
                if isinstance(sub_value, float) and math.isnan(sub_value):
                    raise mlrun.errors.MLRunInvalidArgumentError(nan_error_message)


class KafkaParameters:
    def __init__(self, kwargs: typing.Optional[dict] = None):
        import kafka

        if kwargs is None:
            kwargs = {}
        self._kafka = kafka
        self._kwargs = kwargs
        self._client_configs = {
            "consumer": self._kafka.KafkaConsumer.DEFAULT_CONFIG,
            "producer": self._kafka.KafkaProducer.DEFAULT_CONFIG,
            "admin": self._kafka.KafkaAdminClient.DEFAULT_CONFIG,
        }
        self._custom_attributes = {
            "max_workers": "",
            "brokers": "",
            "topics": "",
            "group": "",
            "initial_offset": "",
            "partitions": "",
            "sasl": "",
            "worker_allocation_mode": "",
        }
        self._reference_dicts = (
            self._custom_attributes,
            self._kafka.KafkaAdminClient.DEFAULT_CONFIG,
            self._kafka.KafkaProducer.DEFAULT_CONFIG,
            self._kafka.KafkaConsumer.DEFAULT_CONFIG,
        )

        self._validate_keys()

    def _validate_keys(self) -> None:
        for key in self._kwargs:
            if all(key not in d for d in self._reference_dicts):
                raise ValueError(
                    f"Key '{key}' not found in any of the Kafka reference dictionaries"
                )

    def _get_config(self, client_type: str) -> dict:
        res = {
            k: self._kwargs[k]
            for k in self._kwargs.keys() & self._client_configs[client_type].keys()
        }
        if sasl := self._kwargs.get("sasl"):
            res |= {
                "security_protocol": "SASL_PLAINTEXT",
                "sasl_mechanism": sasl["mechanism"],
                "sasl_plain_username": sasl["user"],
                "sasl_plain_password": sasl["password"],
            }
        return res

    def consumer(self) -> dict:
        return self._get_config("consumer")

    def producer(self) -> dict:
        return self._get_config("producer")

    def admin(self) -> dict:
        return self._get_config("admin")

    def sasl(
        self, *, usr: typing.Optional[str] = None, pwd: typing.Optional[str] = None
    ) -> dict:
        usr = usr or self._kwargs.get("sasl_plain_username", None)
        pwd = pwd or self._kwargs.get("sasl_plain_password", None)
        res = self._kwargs.get("sasl", {})
        if usr and pwd:
            res["enable"] = True
            res["user"] = usr
            res["password"] = pwd
            res["mechanism"] = self._kwargs.get("sasl_mechanism", "PLAIN")
        return res

    def valid_entries_only(self, input_dict: dict) -> dict:
        valid_keys = set()
        for ref_dict in self._reference_dicts:
            valid_keys.update(ref_dict.keys())
        # Return a new dictionary with only valid keys
        return {k: v for k, v in input_dict.items() if k in valid_keys}
