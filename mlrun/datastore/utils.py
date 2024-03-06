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
import tarfile
import tempfile
import typing
from urllib.parse import parse_qs, urlparse

import pandas as pd
import semver

import mlrun.datastore


def parse_kafka_url(url: str, bootstrap_servers: list = None) -> tuple[str, list]:
    """Generating Kafka topic and adjusting a list of bootstrap servers.

    :param url:               URL path to parse using urllib.parse.urlparse.
    :param bootstrap_servers: List of bootstrap servers for the kafka brokers.

    :return: A tuple of:
         [0] = Kafka topic value
         [1] = List of bootstrap servers
    """
    bootstrap_servers = bootstrap_servers or []

    # Parse the provided URL into six components according to the general structure of a URL
    url = urlparse(url)

    # Add the network location to the bootstrap servers list
    if url.netloc:
        bootstrap_servers = [url.netloc] + bootstrap_servers

    # Get the topic value from the parsed url
    query_dict = parse_qs(url.query)
    if "topic" in query_dict:
        topic = query_dict["topic"][0]
    else:
        topic = url.path
        topic = topic.lstrip("/")
    return topic, bootstrap_servers


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
    time_column: str = None,
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
