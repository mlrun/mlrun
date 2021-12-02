import os
import pathlib
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

        self._connection = pymysql.connect(
            host=mysql_dsn_data["host"],
            user=mysql_dsn_data["username"],
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )

    def close(self):
        self._connection.close()

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
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='mlrun';"
            )
            if cursor.fetchone()[0] > 0:
                return True
        return False

    def check_db_has_data(self):
        with self._connection.cursor() as cursor:
            for check_table in self.check_tables:
                cursor.execute(f"SELECT COUNT(*) FROM `{check_table}`;")
                if cursor.fetchone()[0] > 0:
                    return True
        return False

    def dump_database_to_file(self, filepath: pathlib.Path):
        with self._connection.cursor() as cursor:
            database_dump = self._get_database_dump(cursor)

        with open(str(filepath), "w") as f:
            f.writelines(database_dump)

    @staticmethod
    def get_dsn() -> str:
        return os.environ.get(MySQLUtil.dsn_env_var, "")

    @staticmethod
    def get_mysql_dsn_data() -> typing.Optional[dict]:
        match = re.match(MySQLUtil.dsn_regex, MySQLUtil.get_dsn())
        if not match:
            return None

        return match.groupdict()

    @staticmethod
    def _get_database_dump(cursor) -> str:
        cursor.execute("SHOW TABLES")
        data = ""
        table_names = []
        for table_name in cursor.fetchall():
            table_names.append(table_name[0])

        for table_name in table_names:
            data += f"DROP TABLE IF EXISTS `{table_name}`;"

            cursor.execute(f"SHOW CREATE TABLE `{table_name}`;")
            table_definition = cursor.fetchone()[1]
            data += f"\n{table_definition};\n\n"

            cursor.execute(f"SELECT * FROM `{table_name}`;")
            for row in cursor.fetchall():
                values = ", ".join([f'"{field}"' for field in row])
                data += f"INSERT INTO `{table_name}` VALUES({values});\n"
            data += "\n\n"

        return data
