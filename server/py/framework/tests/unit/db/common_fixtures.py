# Copyright 2024 Iguazio
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
from tempfile import NamedTemporaryFile

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from mlrun.common.db.sql_session import _init_engine
from mlrun.config import config

from framework.db import close_session, create_session
from framework.db.sqldb.db import SQLDB
from framework.tests.unit.common_fixtures import TestServiceBase
from framework.utils.singletons.db import initialize_db
from framework.utils.singletons.project_member import initialize_project_member
from services.api.initial_data import init_data


class TestDatabaseBase(TestServiceBase):
    """
    This fixture initializes a sqlite DB for all tests in the class that inherit from this class.
    Example:
        class TestSomething(TestDatabaseBase):
            # Automatically uses the DB
            def test_something(self):
                ...
    """

    @pytest.fixture(autouse=True)
    def db(self):
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
            # TODO: init data initializes the tables, we should remove this coupling with the API service code
            init_data()
            initialize_project_member()
            self._db = db
            yield
        finally:
            close_session(db_session)

    @pytest.fixture(autouse=True)
    def db_session(self):
        db_session = create_session()
        try:
            self._db_session = db_session
            yield
        finally:
            close_session(db_session)
