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
from urllib.parse import urlparse

from ..config import config
from ..platforms import add_or_refresh_credentials
from .base import RunDBError, RunDBInterface  # noqa
from .filedb import FileRunDB
from .httpdb import HTTPRunDB
from .sqldb import SQLDB
from os import environ


def get_or_set_dburl(default=""):
    if not config.dbpath and default:
        config.dbpath = default
        environ["MLRUN_DBPATH"] = default
    return config.dbpath


def get_httpdb_kwargs(host, username, password):
    username = username or config.httpdb.user
    password = password or config.httpdb.password

    username, password, token = add_or_refresh_credentials(
        host, username, password, config.httpdb.token
    )

    return {
        "user": username,
        "password": password,
        "token": token,
    }


def get_run_db(url=""):
    """Returns the runtime database"""
    if not url:
        url = get_or_set_dburl("./")

    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    kwargs = {}
    if "://" not in url or scheme in ["file", "s3", "v3io", "v3ios"]:
        cls = FileRunDB
    elif scheme in ("http", "https"):
        cls = HTTPRunDB
        kwargs = get_httpdb_kwargs(
            parsed_url.hostname, parsed_url.username, parsed_url.password
        )
        endpoint = parsed_url.hostname
        if parsed_url.port:
            endpoint += ":{}".format(parsed_url.port)
        url = f"{parsed_url.scheme}://{endpoint}{parsed_url.path}"
    else:
        cls = SQLDB

    return cls(url, **kwargs)
