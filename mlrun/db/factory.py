# Copyright 2023 MLRun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from urllib.parse import urlparse

from dependency_injector import containers, providers

import mlrun.db
import mlrun.utils.singleton
from mlrun.config import config
from mlrun.db.httpdb import HTTPRunDB
from mlrun.db.nopdb import NopDB
from mlrun.platforms import add_or_refresh_credentials
from mlrun.utils import logger


class RunDBFactory(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self):
        self._run_db = None
        self._last_db_url = None
        self._rundb_container = RunDBContainer()

    def get_run_db(self, url="", secrets=None, force_reconnect=False):
        """Returns the runtime database"""
        if not url:
            url = mlrun.db.get_or_set_dburl("./")

        if (
            self._last_db_url is not None
            and url == self._last_db_url
            and self._run_db
            and not force_reconnect
        ):
            return self._run_db

        self._last_db_url = url

        kwargs = {}
        if "://" not in str(url):
            logger.warning(
                "Could not detect path to API server, not connected to API server!"
            )
            logger.warning(
                "MLRUN_DBPATH is not set. Set this environment variable to the URL of the API server"
                " in order to connect"
            )
            self._run_db = self._rundb_container.nop(url, **kwargs)

        else:
            self._rundb_container.validate_run_db_url(url)
            url, kwargs = self._rundb_container.resolve_run_db_kwargs(url)
            self._run_db = self._rundb_container.run_db(url, **kwargs)

        self._run_db.connect(secrets=secrets)
        return self._run_db


class RunDBContainer(containers.DeclarativeContainer):
    nop = providers.Factory(NopDB)
    run_db = providers.Factory(HTTPRunDB)

    @staticmethod
    def validate_run_db_url(url):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Invalid scheme {scheme} for MLRUN_DBPATH, only http(s) is supported"
            )

    @staticmethod
    def resolve_run_db_kwargs(url):
        parsed_url = urlparse(url)
        kwargs = RunDBContainer._get_httpdb_kwargs(
            parsed_url.hostname, parsed_url.username, parsed_url.password
        )
        endpoint = parsed_url.hostname
        if parsed_url.port:
            endpoint += f":{parsed_url.port}"
        url = f"{parsed_url.scheme}://{endpoint}{parsed_url.path}"
        return url, kwargs

    @staticmethod
    def _get_httpdb_kwargs(host, username, password):
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
