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
            logger.warn(
                f"Invalid mysql dsn: {MySQLUtil.get_dsn()}, assuming sqlite and skipping liveness verification"
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
