import os
import pathlib
import shutil
import typing

import alembic.command
import alembic.config

from mlrun import mlconf
from mlrun.utils import logger

from .mysql import MySQLUtil


class AlembicUtil(object):
    def __init__(
        self, alembic_config_path: pathlib.Path, data_version_is_latest: bool = True
    ):
        self._alembic_config_path = str(alembic_config_path)
        self._alembic_config = alembic.config.Config(self._alembic_config_path)
        self._alembic_output = ""
        self._data_version_is_latest = data_version_is_latest
        self._db_file_path = self._get_db_file_path()
        # call to _get_current_revision might create dummy db file so we first of all check whether the db file exist
        self._db_path_exists = os.path.isfile(self._db_file_path)
        self._revision_history = self._get_revision_history_list()
        self._latest_revision = self._revision_history[0]
        self._initial_revision = self._revision_history[-1]

    def init_alembic(self, use_backups: bool = False):
        current_revision = self._get_current_revision()

        if (
            use_backups
            and self._db_path_exists
            and current_revision
            and current_revision not in self._revision_history
        ):
            self._downgrade_to_revision(
                self._db_file_path, current_revision, self._latest_revision
            )

        # get current revision again if it changed during the last commands
        current_revision = self._get_current_revision()
        if use_backups and current_revision:
            self._backup_revision(
                self._db_file_path, current_revision, self._latest_revision
            )
        logger.debug("Performing schema migrations")
        alembic.command.upgrade(self._alembic_config, "head")

    def is_schema_migration_needed(self):
        current_revision = self._get_current_revision()
        return current_revision != self._latest_revision

    def is_migration_from_scratch(self):
        current_revision = self._get_current_revision()
        return current_revision != self._initial_revision

    @staticmethod
    def _get_db_file_path() -> str:
        """
        Get the db file path from the dsn.
        Converts the dsn to the file path. e.g.:
        sqlite:////mlrun/db/mlrun.db?check_same_thread=false -> /mlrun/db/mlrun.db
        if mysql is used returns empty string
        """
        if "mysql" in mlconf.httpdb.dsn:
            return ""
        return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]

    def _get_current_revision(self) -> typing.Optional[str]:

        # create separate config in order to catch the stdout
        catch_stdout_config = alembic.config.Config(self._alembic_config_path)
        catch_stdout_config.print_stdout = self._save_output

        self._flush_output()
        try:
            alembic.command.current(catch_stdout_config)
            return self._alembic_output.strip().replace(" (head)", "")
        except Exception as exc:
            if "Can't locate revision identified by" in exc.args[0]:

                # DB has a revision that isn't known to us, extracting it from the exception.
                return exc.args[0].split("'")[2]

            return None

    def _get_revision_history_list(self) -> typing.List[str]:
        """
        Returns a list of the revision history sorted from latest to oldest.
        """

        # create separate config in order to catch the stdout
        catch_stdout_config = alembic.config.Config(self._alembic_config_path)
        catch_stdout_config.print_stdout = self._save_output

        self._flush_output()
        alembic.command.history(catch_stdout_config)
        return self._parse_revision_history(self._alembic_output)

    @staticmethod
    def _parse_revision_history(output: str) -> typing.List[str]:
        return [line.split(" ")[2].replace(",", "") for line in output.splitlines()]

    def _backup_revision(
        self, db_file_path: str, current_version: str, latest_revision: str
    ):
        if db_file_path == ":memory:":
            return

        if self._data_version_is_latest and not self.is_schema_migration_needed():
            logger.debug(
                "Schema version and Data version are latest, skipping backup..."
            )
            return

        if "mysql" in mlconf.httpdb.dsn:
            self._backup_revision_mysql(db_file_path, current_version)
        else:
            self._backup_revision_sqlite(db_file_path, current_version)

    @staticmethod
    def _backup_revision_sqlite(db_file_path: str, current_version: str):
        db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        backup_path = db_dir_path / f"{current_version}.db"

        logger.debug(
            "Backing up DB file", db_file_path=db_file_path, backup_path=backup_path
        )
        shutil.copy2(db_file_path, backup_path)

    @staticmethod
    def _backup_revision_mysql(db_file_path: str, current_version: str):
        db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        backup_path = db_dir_path / f"{current_version}.db"

        mysql_util = MySQLUtil()
        mysql_util.dump_database_to_file(backup_path)

    @staticmethod
    def _downgrade_to_revision(
        db_file_path: str, current_revision: str, fallback_version: str
    ):
        db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        backup_path = db_dir_path / f"{fallback_version}.db"

        if not os.path.isfile(backup_path):
            raise RuntimeError(
                f"Cannot fall back to revision {fallback_version}, "
                f"no back up exists. Current revision: {current_revision}"
            )

        # backup the current DB
        current_backup_path = db_dir_path / f"{current_revision}.db"
        shutil.copy2(db_file_path, current_backup_path)

        shutil.copy2(backup_path, db_file_path)

    def _save_output(self, text: str, *_):
        self._alembic_output += f"{text}\n"

    def _flush_output(self):
        self._alembic_output = ""
