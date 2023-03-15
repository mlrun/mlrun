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
from os import environ
from urllib.parse import urlparse

from ..config import config
from ..platforms import add_or_refresh_credentials
from ..utils import logger
from .base import RunDBError, RunDBInterface  # noqa
from .filedb import FileRunDB
from .sqldb import SQLDB


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


_run_db = None
_last_db_url = None


def get_run_db(url="", secrets=None, force_reconnect=False):
    """Returns the runtime database"""
    global _run_db, _last_db_url

    if not url:
        url = get_or_set_dburl("./")

    if (
        _last_db_url is not None
        and url == _last_db_url
        and _run_db
        and not force_reconnect
    ):
        return _run_db
    _last_db_url = url

    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    kwargs = {}
    if "://" not in str(url) or scheme in ["file", "s3", "v3io", "v3ios"]:
        logger.warning(
            "Could not detect path to API server, Using Deprecated client interface"
        )
        logger.warning(
            "Please make sure your env variable MLRUN_DBPATH is configured correctly to point to the API server!"
        )
        cls = FileRunDB
    elif scheme in ("http", "https"):
        # import here to avoid circular imports
        from .httpdb import HTTPRunDB

        cls = HTTPRunDB
        kwargs = get_httpdb_kwargs(
            parsed_url.hostname, parsed_url.username, parsed_url.password
        )
        endpoint = parsed_url.hostname
        if parsed_url.port:
            endpoint += f":{parsed_url.port}"
        url = f"{parsed_url.scheme}://{endpoint}{parsed_url.path}"
    else:
        cls = SQLDB

    _run_db = cls(url, **kwargs)
    _run_db.connect(secrets=secrets)
    return _run_db
