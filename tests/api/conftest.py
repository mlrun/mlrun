from tempfile import NamedTemporaryFile
from typing import Generator

import pytest

from mlrun.api.db.sqldb.session import create_session, _init_engine
from mlrun.api.initial_data import init_data
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.config import config
from mlrun.utils import logger


@pytest.fixture()
def db() -> Generator:
    """
    This fixture initialize the db singleton (so it will be accessible using mlrun.api.singletons.get_db()
    and generates a db session that can be used by the test
    """
    db_file = NamedTemporaryFile(suffix="-mlrun.db")
    logger.info(f"Created temp db file: {db_file.name}")
    config.httpdb.db_type = "sqldb"
    config.httpdb.dsn = f"sqlite:///{db_file.name}?check_same_thread=false"

    # TODO: make it simpler - doesn't make sense to call 3 different functions to initialize the db
    # we need to force re-init the engine cause otherwise it is cached between tests
    _init_engine(config.httpdb.dsn)
    init_data()
    initialize_db()
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()
