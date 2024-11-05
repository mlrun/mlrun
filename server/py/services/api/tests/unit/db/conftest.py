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
#
from collections.abc import Generator
from tempfile import NamedTemporaryFile

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from mlrun.common.db.sql_session import _init_engine
from mlrun.config import config
from services.api.db.session import close_session, create_session
from services.api.db.sqldb.db import SQLDB
from services.api.initial_data import init_data
from services.api.utils.singletons.db import initialize_db
from services.api.utils.singletons.project_member import initialize_project_member


@pytest.fixture()
def db() -> Generator:
    db_file = NamedTemporaryFile(suffix="-mlrun.db")
    dsn = f"sqlite:///{db_file.name}?check_same_thread=false"
    config.httpdb.dsn = dsn
    _init_engine()

    # SQLite foreign keys constraint must be enabled manually to allow cascade deletions on DB level
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # memory sqldb removes itself when all sessions closed, this session will keep it up until the end of the test
    db_session = create_session()
    try:
        db = SQLDB(dsn)
        db.initialize(db_session)
        initialize_db(db)
        init_data()
        initialize_project_member()
        yield db
    finally:
        close_session(db_session)


@pytest.fixture()
def db_session() -> Generator:
    db_session = create_session()
    try:
        yield db_session
    finally:
        close_session(db_session)
