# Copyright 2018 Iguazio
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
import shutil
from typing import Generator

import pytest

from mlrun.api.db.filedb.db import FileDB
from mlrun.api.db.session import close_session, create_session
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import _init_engine
from mlrun.api.initial_data import init_data
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.api.utils.singletons.project_member import initialize_project_member
from mlrun.config import config

dbs = [
    "sqldb",
    "filedb",
]


@pytest.fixture(params=dbs)
def db(request) -> Generator:
    if request.param == "sqldb":
        dsn = "sqlite:///:memory:?check_same_thread=false"
        config.httpdb.dsn = dsn
        _init_engine()

        # memory sqldb remove it self when all session closed, this session will keep it up during all test
        db_session = create_session()
        try:
            init_data()
            db = SQLDB(dsn)
            db.initialize(db_session)
            initialize_db(db)
            initialize_project_member()
            yield db
        finally:
            close_session(db_session)
    elif request.param == "filedb":
        db = FileDB(config.httpdb.dirpath)
        db_session = create_session(request.param)
        try:
            db.initialize(db_session)

            yield db
        finally:
            shutil.rmtree(config.httpdb.dirpath, ignore_errors=True, onerror=None)
            close_session(db_session)
    else:
        raise Exception("Unknown db type")


@pytest.fixture()
def data_migration_db(request) -> Generator:
    # Data migrations performed before the API goes up, therefore there's no project member yet
    # that's the only difference between this fixture and the db fixture. because of the parameterization it was hard to
    # share code between them, we anyway going to remove filedb soon, then there won't be params, and we could re-use
    # code
    # TODO: fix duplication
    if request.param == "sqldb":
        dsn = "sqlite:///:memory:?check_same_thread=false"
        config.httpdb.dsn = dsn
        _init_engine()

        # memory sqldb remove it self when all session closed, this session will keep it up during all test
        db_session = create_session()
        try:
            init_data()
            db = SQLDB(dsn)
            db.initialize(db_session)
            initialize_db(db)
            yield db
        finally:
            close_session(db_session)
    elif request.param == "filedb":
        db = FileDB(config.httpdb.dirpath)
        db_session = create_session(request.param)
        try:
            db.initialize(db_session)

            yield db
        finally:
            shutil.rmtree(config.httpdb.dirpath, ignore_errors=True, onerror=None)
            close_session(db_session)
    else:
        raise Exception("Unknown db type")


@pytest.fixture(params=dbs)
def db_session(request) -> Generator:
    db_session = create_session(request.param)
    try:
        yield db_session
    finally:
        close_session(db_session)
