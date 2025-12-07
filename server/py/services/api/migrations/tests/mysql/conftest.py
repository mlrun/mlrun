# Copyright 2025 Iguazio
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
from collections.abc import Generator

import pytest
import sqlalchemy
import sqlalchemy.orm

import mlrun

import framework.utils.db.utils
import framework.utils.singletons.db


# TODO: Remove me once we squash all old alembic revisions that create FKs
# that reference non-unique columns.
def _set_mysql_session_variables(dbapi_connection, connection_record):
    """
    Event listener to set MySQL session variables on every new connection.
    This ensures FK constraints can reference non-unique columns (MySQL 8.4+).
    """
    cursor = dbapi_connection.cursor()
    try:
        # MySQL 8.4+ requires unique keys for FK references by default.
        # Disable this restriction at session level for each connection.
        cursor.execute("SET SESSION restrict_fk_on_non_standard_key = OFF")
    finally:
        cursor.close()


@pytest.fixture(scope="session")
def alembic_engine(
    _mysql_engine: sqlalchemy.engine.Engine,
) -> sqlalchemy.engine.Engine:
    """
    Engine bound to the MySQL container – used by pytest-alembic's
    `alembic_runner` fixture.
    """
    # Ensure every new connection gets the relaxed FK session variable.
    sqlalchemy.event.listen(
        _mysql_engine,
        "connect",
        _set_mysql_session_variables,
    )
    # Close any existing connections so subsequent ones get the listener.
    _mysql_engine.dispose()

    os.environ["MLRUN_HTTPDB__DSN"] = _mysql_engine.url.render_as_string(
        hide_password=False,
    )
    mlrun.mlconf.reload()
    framework.utils.singletons.db.initialize_db()
    return _mysql_engine.execution_options(isolation_level="AUTOCOMMIT")


@pytest.fixture(scope="function")
def alembic_session(
    _mysql_engine: sqlalchemy.engine.Engine,
) -> Generator[sqlalchemy.orm.Session, None, None]:
    """
    Real SQLAlchemy *Session* object expected by
    test_notification_params_to_secret_params & friends.
    """
    session_maker = sqlalchemy.orm.sessionmaker(bind=_mysql_engine)
    session = session_maker()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def db_util(
    mysql_db_session: sqlalchemy.orm.Session,
) -> framework.utils.db.utils.DBUtil:
    util = framework.utils.db.utils.DBUtil()
    util.wait_for_db_liveness()
    return util
