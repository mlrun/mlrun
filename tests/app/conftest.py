import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mlrun.api.db.sqldb.session import SessionLocal
from mlrun.api.main import app
from mlrun.api.initial_data import main


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
