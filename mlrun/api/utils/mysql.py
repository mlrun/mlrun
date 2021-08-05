import pathlib
import re
import typing

import pymysql

from mlrun import mlconf


class MySQLUtil(object):
    dsn_regex = (
        r"mysql\+pymysql://(?P<username>.+)@(?P<host>.+):(?P<port>\d+)/(?P<database>.+)"
    )
    check_table = 'projects'

    def __init__(self):
        mysql_dsn_data = self.get_mysql_dsn_data()
        if not mysql_dsn_data:
            raise RuntimeError(f"Invalid mysql dsn: {mlconf.httpdb.dsn}")

        self._connection = pymysql.connect(
            host=mysql_dsn_data["host"],
            user=mysql_dsn_data["username"],
            port=int(mysql_dsn_data["port"]),
            database=mysql_dsn_data["database"],
        )

    def check_db_has_data(self):
        with self._connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM `{self.check_table}`;")
            return cursor.fetchone()[0] > 0

    def dump_database_to_file(self, filepath: pathlib.Path):
        with self._connection.cursor() as cursor:
            database_dump = self._get_database_dump(cursor)

        with open(str(filepath), "w") as f:
            f.writelines(database_dump)

    @staticmethod
    def get_mysql_dsn_data() -> typing.Optional[dict]:
        match = re.match(MySQLUtil.dsn_regex, mlconf.httpdb.dsn)
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
