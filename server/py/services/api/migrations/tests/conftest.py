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
import os
import typing

import pytest
import sqlalchemy
import sqlalchemy.orm

import mlrun.utils
import tests.conftest
import tests.sitecustomize  # noqa: F401 - registers os.register_at_fork for coverage in forked processes

pytest_plugins = [
    "tests.common_fixtures",
    "tests.conftest",
]


logger = mlrun.utils.create_test_logger()
mlrun.utils.logger.get_handler("default").setFormatter(
    mlrun.utils.resolve_formatter_by_kind(mlrun.utils.FormatterKinds.HUMAN_EXTENDED)()
)


def _session_for_engine(
    engine: sqlalchemy.engine.Engine,
) -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    import framework.db.sqldb.models
    import framework.utils.singletons.db

    tests.conftest._wipe_database(engine)
    mlrun.utils.logger.info("Wiping database", db_type=engine.name)
    framework.db.sqldb.models.Base.metadata.create_all(engine)
    framework.utils.singletons.db.initialize_db()

    session = sqlalchemy.orm.sessionmaker(bind=engine)()
    yield session


@pytest.fixture(scope="session")
def postgres_db_session(
    _postgres_engine: sqlalchemy.engine.Engine,
    request: pytest.FixtureRequest,
) -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """Session bound to the Postgres test DB."""

    mlrun.utils.logger.info("Starting database engine", db_type="postgres")
    os.environ["MLRUN_HTTPDB__DSN"] = _postgres_engine.url.render_as_string(
        hide_password=False
    )
    mlrun.mlconf.reload()
    yield from _session_for_engine(_postgres_engine)


@pytest.fixture(scope="session")
def mysql_db_session(
    _mysql_engine: sqlalchemy.engine.Engine,
    request: pytest.FixtureRequest,
) -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """Session bound to the MySQL test DB."""

    mlrun.utils.logger.info("Starting database engine", db_type="mysql")
    os.environ["MLRUN_HTTPDB__DSN"] = _mysql_engine.url.render_as_string(
        hide_password=False
    )
    mlrun.mlconf.reload()
    yield from _session_for_engine(
        engine=_mysql_engine,
    )


@pytest.fixture
def db_engine_with_schema(
    db_engine: sqlalchemy.engine.Engine,
) -> typing.Generator[sqlalchemy.engine.Engine, None, None]:
    """Start a DB engine according to ``MLRUN_TEST_DB`` and yield it."""
    import framework.db.sqldb.models
    import framework.utils.singletons.db

    framework.db.sqldb.models.Base.metadata.drop_all(db_engine)
    framework.db.sqldb.models.Base.metadata.create_all(db_engine)
    framework.utils.singletons.db.initialize_db()
    yield db_engine


@pytest.fixture
def alembic_session(
    db_engine: sqlalchemy.engine.Engine,
) -> typing.Generator[sqlalchemy.orm.Session, None, None]:
    """Plain SQLAlchemy session bound to *alembic_engine* for migration tests."""

    session_class = sqlalchemy.orm.sessionmaker(bind=db_engine)
    session = session_class()
    try:
        yield session
    finally:
        session.close()
