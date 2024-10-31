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
import os
import re
import typing

import pymysql

import mlrun.utils
from mlrun.config import config as mlconf


class MySQLUtil:
    dsn_env_var = "MLRUN_HTTPDB__DSN"
    dsn_regex = r"mysql\+pymysql://(?P<username>[^:]+)(?::(?P<password>[^@]*))?@(?P<host>[^:]+)(?::(?P<port>\d+))?/(?P<database>.+)"
    check_tables = [
        "projects",
        # check functions as well just in case the previous version used a projects leader
        "functions",
    ]

    def __init__(self, logger: mlrun.utils.Logger):
        self._logger = logger

    def wait_for_db_liveness(self, retry_interval=3, timeout=2 * 60):
        self._logger.debug("Waiting for database liveness")
        mysql_dsn_data = self.get_mysql_dsn_data()
        tmp_connection = mlrun.utils.retry_until_successful(
            retry_interval,
            timeout,
            self._logger,
            True,
            pymysql.connect,
            host=mysql_dsn_data["host"],
            user=mysql_dsn_data["username"],
            password=mysql_dsn_data["password"],
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )
        self._logger.debug("Database ready for connection")
        tmp_connection.close()

    def check_db_has_tables(self):
        connection = self._create_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='mlrun';"
                )
                if cursor.fetchone()[0] > 0:
                    return True
            return False
        finally:
            connection.close()

    def set_modes(self, modes):
        if not modes or modes in ["nil", "none"]:
            self._logger.debug("No sql modes were given, bailing", modes=modes)
            return
        connection = self._create_connection()
        try:
            self._logger.debug("Setting sql modes", modes=modes)
            with connection.cursor() as cursor:
                cursor.execute("SET GLOBAL sql_mode=%s;", (modes,))
        finally:
            connection.close()

    def check_db_has_data(self):
        connection = self._create_connection()
        try:
            with connection.cursor() as cursor:
                for check_table in self.check_tables:
                    cursor.execute("SELECT COUNT(*) FROM %s;", (check_table,))
                    if cursor.fetchone()[0] > 0:
                        return True
            return False
        finally:
            connection.close()

    def _create_connection(self):
        mysql_dsn_data = self.get_mysql_dsn_data()
        if not mysql_dsn_data:
            raise RuntimeError(f"Invalid mysql dsn: {self.get_dsn()}")
        return pymysql.connect(
            host=mysql_dsn_data["host"],
            user=mysql_dsn_data["username"],
            password=mysql_dsn_data["password"],
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )

    @staticmethod
    def get_mysql_dsn_data() -> typing.Optional[dict]:
        match = re.match(MySQLUtil.dsn_regex, MySQLUtil.get_dsn())
        if not match:
            return None

        return match.groupdict()

    @staticmethod
    def get_dsn() -> str:
        return os.environ.get(MySQLUtil.dsn_env_var, mlconf.httpdb.dsn)
