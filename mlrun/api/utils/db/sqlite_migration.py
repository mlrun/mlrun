import typing

import sqlite3_to_mysql

from mlrun import mlconf
from mlrun.utils import logger

from .mysql import MySQLUtil


class SQLiteMigrationUtil(object):
    def __init__(self):
        self._mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        self._migrator = self._create_migrator()
        self._mysql_util = None
        if self._mysql_dsn_data:
            self._mysql_util = MySQLUtil()

    def transfer(self):

        # if some data is missing, don't transfer the data
        if not self._migrator:
            return

        db_has_data = False
        if self._mysql_util:
            if self._mysql_util.check_db_has_data():
                db_has_data = True
            self._mysql_util.close()

        # if mysqldb already has data, don't transfer the data
        if db_has_data:
            return

        self._migrator.transfer()

    def _create_migrator(self) -> typing.Optional[sqlite3_to_mysql.SQLite3toMySQL]:
        sqlite_file = self._get_old_db_file_path()
        if not sqlite_file or not self._mysql_dsn_data:
            return None

        logger.info(
            "Creating SQLite to MySQL Converter",
            sqlite_file=sqlite_file,
            mysql_dsn_data=self._mysql_dsn_data,
        )

        return sqlite3_to_mysql.SQLite3toMySQL(
            sqlite_file=sqlite_file,
            mysql_user=self._mysql_dsn_data["username"],
            mysql_database=self._mysql_dsn_data["database"],
            mysql_host=self._mysql_dsn_data["host"],
            mysql_port=int(self._mysql_dsn_data["port"]),
            quiet=True,
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
        return mlconf.httpdb.old_dsn.split("?")[0].split("sqlite:///")[-1]
