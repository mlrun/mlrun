import os.path
import typing

import sqlite3_to_mysql

from mlrun import mlconf
from mlrun.utils import logger

from .mysql import MySQLUtil


class SQLiteMigrationUtil(object):

    # All tables except 'alembic_version' so the sqlite migration tree doesn't override the mysql migration tree
    sqlite_tables = [
        "artifacts",
        "functions",
        "logs",
        "runs",
        "schedules",
        "users",
        "projects",
        "artifacts_labels",
        "artifacts_tags",
        "functions_labels",
        "functions_tags",
        "runs_labels",
        "runs_tags",
        "project_users",
        "schedules_v2",
        "schedules_v2_labels",
        "entities",
        "feature_sets_labels",
        "feature_sets_tags",
        "features",
        "entities_labels",
        "features_labels",
        "feature_sets",
        "feature_vectors",
        "feature_vectors_labels",
        "feature_vectors_tags",
        "projects_labels",
        "data_versions",
    ]

    def __init__(self):
        self._mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        self._migrator = self._create_migrator()
        self._mysql_util = None
        if self._mysql_dsn_data:
            self._mysql_util = MySQLUtil()

    def is_database_migration_needed(self) -> bool:
        # if some data is missing, don't transfer the data
        if not self._migrator:
            return False

        db_has_data = False
        if self._mysql_util:
            if (
                self._mysql_util.check_db_has_tables()
                and self._mysql_util.check_db_has_data()
            ):
                db_has_data = True

        # if mysqldb already has data, don't transfer the data
        if db_has_data:
            return False
        return True

    def transfer(self):
        if not self.is_database_migration_needed():
            return

        self._migrator.transfer()

    def _create_migrator(self) -> typing.Optional[sqlite3_to_mysql.SQLite3toMySQL]:
        sqlite_file = self._get_old_db_file_path()
        if (
            not sqlite_file
            or not self._mysql_dsn_data
            or not os.path.isfile(sqlite_file)
        ):
            return None

        logger.info(
            "Creating SQLite to MySQL Converter",
            sqlite_file=sqlite_file,
            mysql_dsn_data=self._mysql_dsn_data,
        )

        return sqlite3_to_mysql.SQLite3toMySQL(
            sqlite_file=sqlite_file,
            sqlite_tables=self.sqlite_tables,
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
