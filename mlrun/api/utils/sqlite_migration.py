import typing

import sqlite3_to_mysql

from mlrun import mlconf

from .mysql import MySQLUtil


class SQLiteMigrationUtil(object):
    def __init__(self):
        self._converter = self._create_converter()

    def transfer(self):
        if not self._converter:
            return
        self._converter.transfer()

    def _create_converter(self) -> typing.Optional[sqlite3_to_mysql.SQLite3toMySQL]:
        sqlite_file = self._get_old_db_file_path()
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if not sqlite_file or not mysql_dsn_data:
            return None

        return sqlite3_to_mysql.SQLite3toMySQL(
            sqlite_file=sqlite_file,
            mysql_user=mysql_dsn_data["username"],
            mysql_database=mysql_dsn_data["database"],
            mysql_host=mysql_dsn_data["host"],
            mysql_port=int(mysql_dsn_data["port"]),
        )

    @staticmethod
    def _get_old_db_file_path() -> str:
        """
        Get the db file path from the old_dsn.
        Converts the dsn to the file path. e.g.:
        sqlite:////mlrun/db/mlrun.db?check_same_thread=false -> /mlrun/db/mlrun.db
        """
        if not mlconf.httpdb.old_dsn:
            return ""
        return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]
