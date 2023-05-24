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
#
import tarfile
import tempfile
import typing
from urllib.parse import parse_qs, urlparse

import pandas as pd

import mlrun.datastore


def store_path_to_spark(path):
    if path.startswith("redis://") or path.startswith("rediss://"):
        url = urlparse(path)
        if url.path:
            path = url.path
    elif path.startswith("v3io:///"):
        path = "v3io:" + path[len("v3io:/") :]
    elif path.startswith("s3://"):
        if path.startswith("s3:///"):
            # 's3:///' not supported since mlrun 0.9.0 should use s3:// instead
            from mlrun.errors import MLRunInvalidArgumentError

            valid_path = "s3:" + path[len("s3:/") :]
            raise MLRunInvalidArgumentError(
                f"'s3:///' is not supported, try using 's3://' instead.\nE.g: '{valid_path}'"
            )
        else:
            path = "s3a:" + path[len("s3:") :]
    return path


def parse_kafka_url(
    url: str, bootstrap_servers: typing.List = None
) -> typing.Tuple[str, typing.List]:
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
        datastore, subpath = stores.get_or_create_store(target)
        datastore.upload(subpath, temp_fh.name)


class RestartableIterator:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)


def filter_df_start_end_time(df, time_field=None, start_time=None, end_time=None):
    df = _iter_over_df(df)
    for df_in in df:
        if time_field:
            df_in[time_field] = pd.to_datetime(df_in[time_field])
            if start_time:
                df_in = df_in[df_in[time_field] > start_time]
            if end_time:
                df_in = df_in[df_in[time_field] <= end_time]
            if not isinstance(df, RestartableIterator):
                return df_in
        return df


def select_columns_from_df(df, columns):
    df = _iter_over_df(df)
    for df_in in df:
        if columns:
            df_in = df_in[columns]
        if not isinstance(df, RestartableIterator):
            return df_in
    return df


def _iter_over_df(df):
    if not isinstance(df, pd.DataFrame):
        df = RestartableIterator(df)
    else:
        df = [df]
    return df
