import os
import alembic.config
import alembic.command

from mlrun import mlconf


INITIAL_ALEMBIC_REVISION = "11f8dd2dc9fe"


def init_alembic():
    if os.path.isfile(_get_db_file_path()) and not _get_current_database_version():

        # if database file exists but no alembic version exists, stamp the existing
        # database with the initial alembic version, so we can upgrade it later
        run_alembic_command("stamp", INITIAL_ALEMBIC_REVISION)

    run_alembic_command("upgrade", "head")


def run_alembic_command(*args):
    # raise error to exit on a failure
    argv = ["--raiseerr"]
    argv.extend(args)
    alembic.config.main(argv=argv)


def _get_db_file_path():
    return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]


def _get_current_database_version():
    alembic_cfg = alembic.config.Config("alembic.ini")

    captured_text = None

    def print_stdout(text, *arg):
        nonlocal captured_text
        captured_text = text

    alembic_cfg.print_stdout = print_stdout
    alembic.command.current(alembic_cfg)
    return captured_text
