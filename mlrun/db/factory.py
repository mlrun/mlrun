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

    def create_run_db(
        self, url="", secrets=None, force_reconnect=False, **kwargs
    ) -> mlrun.db.RunDBInterface:
        """
        Returns the runtime database
        :param url:             DB URL to connect to
        :param secrets:         DB connection secrets
        :param force_reconnect: Force reconnect to the DB
        :param kwargs:          Extra arguments to pass to the Run DB constructor
        :return:                Run DB implementation of :py:class:`~mlrun.db.RunDBInterface`
        """
        if not url:
            url = mlrun.db.get_or_set_dburl("./")

        if url != self._last_db_url:
            # if the url changed, we need to reconnect
            force_reconnect = True

        self._last_db_url = url

        if "://" not in str(url):
            logger.warning(
                "Could not detect path to API server, not connected to API server!"
            )
            logger.warning(
                "MLRUN_DBPATH is misconfigured. Set this environment variable to the URL of the API server"
                " in order to connect"
            )
            factory = self._rundb_container.nop

        elif url.startswith("http"):
            if force_reconnect:
                # httpdb is a singleton, so we need to reset it in order to reconnect
                self._rundb_container.http_db.reset()

            factory = self._rundb_container.http_db

        else:
            factory = self._rundb_container.sql_db

        return factory(url, secrets=secrets, **kwargs)


class RunDBContainer(containers.DeclarativeContainer):
    nop = providers.Factory(mlrun.db.nopdb.NopDB)
    http_db = providers.Singleton(mlrun.db.httpdb.HTTPRunDB)
    sql_db = providers.Factory(mlrun.db.RunDBInterface)
