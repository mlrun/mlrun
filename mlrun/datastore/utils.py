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
import typing
from urllib.parse import parse_qs, urlparse


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
