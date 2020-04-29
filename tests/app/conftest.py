import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mlrun.app.db.sqldb.session import SessionLocal
from mlrun.app.main import app
from mlrun.app.initial_data import main


@pytest.fixture(scope="session")
def db() -> Generator:
    main()
    yield SessionLocal()

    # TODO: should be dynamic based on configuration
    os.remove("/tmp/mlrun.db")


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c
