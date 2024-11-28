# Copyright 2023 Iguazio
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
from dependency_injector import containers, providers

import mlrun.db
import mlrun.db.httpdb
import mlrun.db.nopdb
import mlrun.utils.singleton
from mlrun.utils import logger


class RunDBFactory(
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    def __init__(self):
        self._run_db = None
        self._last_db_url = None
        self._rundb_container = RunDBContainer()

    def create_run_db(self, url="", secrets=None, force_reconnect=False):
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

        if "://" not in str(url):
            logger.warning(
                "Could not detect path to API server, not connected to API server!"
            )
            logger.warning(
                "MLRUN_DBPATH is misconfigured. Set this environment variable to the URL of the API server"
                " in order to connect"
            )
            self._run_db = self._rundb_container.nop(url)

        else:
            self._run_db = self._rundb_container.run_db(url)

        self._run_db.connect(secrets=secrets)
        return self._run_db


class RunDBContainer(containers.DeclarativeContainer):
    nop = providers.Factory(mlrun.db.nopdb.NopDB)
    run_db = providers.Factory(mlrun.db.httpdb.HTTPRunDB)
