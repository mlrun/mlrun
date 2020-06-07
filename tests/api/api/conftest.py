from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mlrun.api.db.sqldb.session import create_session, _init_engine
from mlrun.api.initial_data import init_data
from mlrun.api.main import app
from mlrun.utils import logger


@pytest.fixture()
def db() -> Generator:
    with NamedTemporaryFile(suffix="-mlrun.db") as db_file:
        logger.info(f"Created temp db file: {db_file.name}")
        _init_engine(f"sqlite:///{db_file.name}?check_same_thread=false")
        init_data()
        yield create_session()


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as c:
        yield c
