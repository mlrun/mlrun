# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import traceback
import typing
from datetime import datetime
from http import HTTPStatus
from os import environ
from pathlib import Path
from subprocess import run
from sys import executable, platform, stderr
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

import pytest
import pytest_mock_resources
import sqlalchemy
import sqlalchemy.orm

import mlrun.common.db.dialects
import mlrun.utils

tests_root_directory = Path(__file__).absolute().parent
results = tests_root_directory / "test_results"
is_ci = "CI" in environ

environ["KFPMETA_OUT_DIR"] = f"{results}/kfp/"
environ["KFP_ARTIFACTS_DIR"] = f"{results}/kfp/"

rundb_path = f"{results}/rundb"
logs_path = f"{results}/logs"
out_path = f"{results}/out"
root_path = str(Path(tests_root_directory).parent)
examples_path = Path(tests_root_directory).parent.joinpath("examples")
pytest_plugins = ["tests.common_fixtures"]

run_time_fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
logging.getLogger("faker.factory").setLevel(logging.WARNING)


def check_docker():
    if not platform.startswith("linux"):
        return False

    with open("/proc/1/cgroup") as fp:
        for line in fp:
            if "/docker/" in line:
                return True
    return False


in_docker = check_docker()

# This must be *after* environment changes above
from mlrun import RunObject, RunTemplate  # noqa
from mlrun.utils import FormatterKinds, create_test_logger, resolve_formatter_by_kind  # noqa

logger = create_test_logger()

logger.get_handler("default").setFormatter(
    resolve_formatter_by_kind(FormatterKinds.HUMAN_EXTENDED)()
)


def tag_test(spec: RunTemplate, name) -> RunTemplate:
    spec = spec.copy()
    spec.metadata.name = name
    spec.metadata.labels["test"] = name
    return spec


def has_secrets():
    return Path("secrets.txt").is_file()


def verify_state(result: RunObject, expected="completed"):
    state = result.status.state
    assert state == expected, f"wrong state ({state}) {result.status.error}"


def wait_for_server(url, timeout_sec):
    start = monotonic()
    while monotonic() - start <= timeout_sec:
        try:
            with urlopen(url) as resp:
                if resp.status == HTTPStatus.OK.value:
                    return True
        except (URLError, ConnectionError):
            pass
        sleep(0.1)
    return False


def run_now():
    return mlrun.utils.format_datetime(datetime.now(), run_time_fmt)


def new_run(state, labels, uid=None, **kw):
    obj = {
        "metadata": {"name": "run-name", "labels": labels},
        "status": {"state": state, "start_time": run_now()},
    }
    if uid:
        obj["metadata"]["uid"] = uid
    obj.update(kw)
    return obj


def freeze(f, **kwargs):
    """
    Enables to override an attribute passed to a sub-function without the need to access the function directly
    :param f: the function we want to pass the attribute to
    :param kwargs: dictionary containing name(key) and value of the attributes to override.
    :return: wrapped function with overridden attributes
    """
    frozen = kwargs

    def wrapper(*args, **kwargs):
        kwargs.update(frozen)
        return f(*args, **kwargs)

    return wrapper


def exec_mlrun(args, cwd=None, op="run"):
    cmd = [executable, "-m", "mlrun", op] + args
    out = run(cmd, capture_output=True, cwd=cwd)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=stderr)
        print(out.stdout.decode("utf-8"), file=stderr)
        print(traceback.format_exc())
        raise Exception(out.stderr.decode("utf-8"))
    return out.stdout.decode("utf-8")


class MockSpecificCalls:
    def __init__(
        self,
        original_function: typing.Callable,
        call_indexes_to_mock: list[int],
        return_value: typing.Any,
    ):
        self.original_function = original_function
        self.call_indexes_to_mock = call_indexes_to_mock
        self.return_value = return_value

    calls = 0

    def mock_function(self, *args, **kwargs):
        self.calls += 1
        if self.calls not in self.call_indexes_to_mock:
            return self.original_function(*args, **kwargs)
        else:
            return self.return_value


# Determine which backend is under test
TEST_DB = os.getenv("MLRUN_TEST_DB", mlrun.common.db.dialects.Dialects.MYSQL)

MYSQL_ONLY_TEST = pytest.mark.skipif(
    not mlrun.common.db.dialects.Dialects.MYSQL.startswith(TEST_DB),
    reason="MySQL-only test",
)
PG_ONLY_TEST = pytest.mark.skipif(
    not mlrun.common.db.dialects.Dialects.POSTGRESQL.startswith(TEST_DB),
    reason="Postgres-only test",
)

_mysql_engine = pytest_mock_resources.create_mysql_fixture(
    scope="session",
)

_postgres_engine = pytest_mock_resources.create_postgres_fixture(
    scope="session",
)


@pytest.fixture(scope="session")
def pmr_mysql_config() -> pytest_mock_resources.MysqlConfig:
    return pytest_mock_resources.MysqlConfig(
        image="mysql:8.4",
        port=3306,
        username="root",
        password="pass",
        root_database="mlrun",
    )


@pytest.fixture(scope="session")
def pmr_postgres_config() -> pytest_mock_resources.PostgresConfig:
    return pytest_mock_resources.PostgresConfig(
        image=os.getenv("MLRUN_POSTGRES_IMAGE", "gcr.io/iguazio/postgres:17"),
        port=5432,
        username="root",
        password="pass",
        root_database="mlrun",
        drivername="postgresql+psycopg2",
    )


def _wipe_database(
    engine: sqlalchemy.engine.Engine,
) -> None:
    """Truncate all user tables & reset sequences."""
    insp = sqlalchemy.inspect(engine)
    with engine.begin() as conn:
        if engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
            tables = insp.get_table_names(schema="public")
            if tables:
                conn.execute(
                    sqlalchemy.text(
                        "DROP TABLE " + ", ".join(f'"{t}"' for t in tables) + " CASCADE"
                    )
                )
        elif engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 0"))
            for t in insp.get_table_names():
                conn.execute(sqlalchemy.text(f"DROP TABLE `{t}`"))
            conn.execute(sqlalchemy.text("SET FOREIGN_KEY_CHECKS = 1"))
        elif engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.SQLITE):
            tables = insp.get_table_names()
            if tables:
                for table in tables:
                    conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table}`"))
        else:
            raise ValueError(f"Unsupported database dialect: {engine.dialect.name}")


@pytest.fixture
def db_engine(
    request: pytest.FixtureRequest,
) -> typing.Generator[sqlalchemy.engine.Engine, None, None]:
    db_type = os.getenv("MLRUN_TEST_DB", "mysql").lower()
    logger.info("Starting database engine", db_type=db_type)

    engine: sqlalchemy.engine.Engine = request.getfixturevalue(
        "_postgres_engine" if db_type == "postgres" else "_mysql_engine"
    )

    logger.info("Started database engine", db_type=db_type)
    os.environ["MLRUN_HTTPDB__DSN"] = engine.url.render_as_string(hide_password=False)
    mlrun.mlconf.reload()
    logger.info("Wiping database", db_type=db_type)
    _wipe_database(engine)
    yield engine
