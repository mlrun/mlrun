from typing import Generator

import pytest
import shutil

from mlrun.api.db.filedb.db import FileDB
from mlrun.api.db.session import create_session, close_session
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import _init_engine
from mlrun.api.initial_data import init_data
from mlrun.config import config

dbs = [
    "sqldb",
    "filedb",
]


@pytest.fixture(params=dbs)
def db(request) -> Generator:
    if request.param == "sqldb":
        dsn = "sqlite:///:memory:?check_same_thread=false"
        _init_engine(dsn)

        # memory sqldb remove it self when all session closed, this session will keep it up during all test
        db_session = create_session()
        try:
            init_data()
            db = SQLDB(dsn)
            db.initialize(db_session)
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
