import os
import alembic.config
import alembic.command

from mlrun import mlconf


class AlembicUtil(object):
    class Constants(object):
        initial_alembic_revision = "11f8dd2dc9fe"

    def __init__(self, alembic_config_path):
        self._alembic_config_path = str(alembic_config_path)
        self._current_revision = None

    def init_alembic(self):
        if (
            os.path.isfile(self._get_db_file_path())
            and not self._get_current_database_version()
        ):

            # if database file exists but no alembic version exists, stamp the existing
            # database with the initial alembic version, so we can upgrade it later
            self.run_alembic_command("stamp", self.Constants.initial_alembic_revision)

        self.run_alembic_command("upgrade", "head")

    def run_alembic_command(self, *args):
        # raise error to exit on a failure
        argv = [
            "--raiseerr",
            "-c",
            f"{self._alembic_config_path}",
        ]
        argv.extend(args)
        alembic.config.main(argv=argv)

    @staticmethod
    def _get_db_file_path():
        return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]

    def _get_current_database_version(self):
        alembic_cfg = alembic.config.Config(self._alembic_config_path)

        def print_stdout(text, *arg):
            self._current_revision = text

        alembic_cfg.print_stdout = print_stdout
        alembic.command.current(alembic_cfg)
        return self._current_revision
