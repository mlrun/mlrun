import typing

import sqlite3_to_mysql

from mlrun import mlconf

from .mysql import MySQLUtil


class SQLiteMigrationUtil(object):
    def __init__(self):
        self._converter = self._create_converter()
        self._mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        self._mysql_util = None
        if self._mysql_dsn_data:
            self._mysql_util = MySQLUtil()

    def transfer(self):

        # if some data is missing, don't transfer the data
        if not self._converter:
            return

        # if mysqldb already has data, don't transfer the data
        if self._mysql_util and self._mysql_util.check_db_has_data():
            return

        self._converter.transfer()

    def _create_converter(self) -> typing.Optional[sqlite3_to_mysql.SQLite3toMySQL]:
        sqlite_file = self._get_old_db_file_path()
        if not sqlite_file or not self._mysql_dsn_data:
            return None

        return sqlite3_to_mysql.SQLite3toMySQL(
            sqlite_file=sqlite_file,
            mysql_user=self._mysql_dsn_data["username"],
            mysql_database=self._mysql_dsn_data["database"],
            mysql_host=self._mysql_dsn_data["host"],
            mysql_port=int(self._mysql_dsn_data["port"]),
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
