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
from urllib.parse import urlparse


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


def parse_kafka_url(url, bootstrap_servers=None):
    bootstrap_servers = bootstrap_servers or []
    url = urlparse(url)
    if url.netloc:
        bootstrap_servers = [url.netloc] + bootstrap_servers
    topic = url.path
    topic = topic.lstrip("/")
    return topic, bootstrap_servers
