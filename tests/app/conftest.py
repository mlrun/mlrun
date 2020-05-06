from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mlrun.api.db.sqldb.session import create_session, _init_engine
from mlrun.api.initial_data import main as init_data
from mlrun.api.main import app
from mlrun.utils import logger


@pytest.fixture(scope="module")
def db() -> Generator:
    db_file = NamedTemporaryFile(suffix="-mlrun.db")
    logger.info(f"Created temp db file: {db_file.name}")
    _init_engine(f"sqlite:///{db_file.name}?check_same_thread=false")
    init_data()
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c
