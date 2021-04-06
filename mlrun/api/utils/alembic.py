import os
import pathlib
import shutil
import typing

import alembic.command
import alembic.config

from mlrun import mlconf


class AlembicUtil(object):
    def __init__(self, alembic_config_path: pathlib.Path):
        self._alembic_config_path = str(alembic_config_path)
        self._alembic_config = alembic.config.Config(self._alembic_config_path)
        self._alembic_output = ""

    def init_alembic(self, from_scratch: bool = False):
        revision_history = self._get_revision_history_list()
        latest_revision = revision_history[0]
        initial_alembic_revision = revision_history[-1]
        db_file_path = self._get_db_file_path()
        db_path_exists = os.path.isfile(db_file_path)
        # this command for some reason creates a dummy db file so it has to be after db_path_exists
        current_revision = self._get_current_revision()

        if not from_scratch and db_path_exists and not current_revision:

            # if database file exists but no alembic version exists, stamp the existing
            # database with the initial alembic version, so we can upgrade it later
            alembic.command.stamp(self._alembic_config, initial_alembic_revision)

        elif (
            db_path_exists
            and current_revision
            and current_revision not in revision_history
        ):
            self._downgrade_to_revision(db_file_path, current_revision, latest_revision)

        # get current revision again if it changed during the last commands
        current_revision = self._get_current_revision()
        if current_revision:
            self._backup_revision(db_file_path, current_revision)
        alembic.command.upgrade(self._alembic_config, "head")

    @staticmethod
    def _get_db_file_path() -> str:
        """
        Get the db file path from the dsn.
        Converts the dsn to the file path. e.g.:
        sqlite:////mlrun/db/mlrun.db?check_same_thread=false -> /mlrun/db/mlrun.db
        """
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

    @staticmethod
    def _backup_revision(db_file_path: str, current_version: str):
        if db_file_path == ":memory:":
            return

        db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        backup_path = db_dir_path / f"{current_version}.db"

        shutil.copy2(db_file_path, backup_path)

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
