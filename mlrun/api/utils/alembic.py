import os
import alembic.config
import alembic.command

from mlrun import mlconf


class AlembicUtil(object):
    class Constants(object):
        initial_alembic_revision = "11f8dd2dc9fe"

    def __init__(self, alembic_config_path):
        self._alembic_config_path = str(alembic_config_path)
        self._alembic_config = alembic.config.Config(self._alembic_config_path)
        self._current_revision = None

    def init_alembic(self):
        if (
            os.path.isfile(self._get_db_file_path())
            and not self._get_current_revision()
        ):

            # if database file exists but no alembic version exists, stamp the existing
            # database with the initial alembic version, so we can upgrade it later
            alembic.command.stamp(
                self._alembic_config, self.Constants.initial_alembic_revision
            )

        alembic.command.upgrade(self._alembic_config, "head")

    @staticmethod
    def _get_db_file_path():
        return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]

    def _get_current_revision(self):
        def print_stdout(text, *arg):
            self._current_revision = text

        # create separate config in order to catch the stdout
        catch_stdout_config = alembic.config.Config(self._alembic_config_path)
        catch_stdout_config.print_stdout = print_stdout

        alembic.command.current(catch_stdout_config)
        return self._current_revision
