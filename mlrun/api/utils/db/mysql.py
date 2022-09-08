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
import os
import re
import typing

import pymysql

import mlrun.utils


class MySQLUtil(object):
    dsn_env_var = "MLRUN_HTTPDB__DSN"
    dsn_regex = (
        r"mysql\+pymysql://(?P<username>.+)@(?P<host>.+):(?P<port>\d+)/(?P<database>.+)"
    )
    check_tables = [
        "projects",
        # check functions as well just in case the previous version used a projects leader
        "functions",
    ]

    def __init__(self):
        mysql_dsn_data = self.get_mysql_dsn_data()
        if not mysql_dsn_data:
            raise RuntimeError(f"Invalid mysql dsn: {self.get_dsn()}")

    @staticmethod
    def wait_for_db_liveness(logger, retry_interval=3, timeout=2 * 60):
        logger.debug("Waiting for database liveness")
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if not mysql_dsn_data:
            dsn = MySQLUtil.get_dsn()
            if "sqlite" in dsn:
                logger.debug("SQLite DB is used, liveness check not needed")
            else:
                logger.warn(
                    f"Invalid mysql dsn: {MySQLUtil.get_dsn()}, assuming live and skipping liveness verification"
                )
            return

        tmp_connection = mlrun.utils.retry_until_successful(
            retry_interval,
            timeout,
            logger,
            True,
            pymysql.connect,
            host=mysql_dsn_data["host"],
            user=mysql_dsn_data["username"],
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )
        logger.debug("Database ready for connection")
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

    def check_db_has_data(self):
        connection = self._create_connection()
        try:
            with connection.cursor() as cursor:
                for check_table in self.check_tables:
                    cursor.execute(f"SELECT COUNT(*) FROM `{check_table}`;")
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
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )

    @staticmethod
    def get_dsn() -> str:
        return os.environ.get(MySQLUtil.dsn_env_var, "")

    @staticmethod
    def get_mysql_dsn_data() -> typing.Optional[dict]:
        match = re.match(MySQLUtil.dsn_regex, MySQLUtil.get_dsn())
        if not match:
            return None

        return match.groupdict()
