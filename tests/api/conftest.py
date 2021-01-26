from tempfile import NamedTemporaryFile
from typing import Generator

import pytest

from mlrun.api.db.sqldb.session import create_session, _init_engine
from mlrun.api.initial_data import init_data
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.api.utils.singletons.project_member import initialize_project_member
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
    dsn = f"sqlite:///{db_file.name}?check_same_thread=false"
    config.httpdb.dsn = dsn

    # TODO: make it simpler - doesn't make sense to call 3 different functions to initialize the db
    # we need to force re-init the engine cause otherwise it is cached between tests
    _init_engine(config.httpdb.dsn)

    # forcing from scratch because we created an empty file for the db
    init_data(from_scratch=True)
    initialize_db()
    initialize_project_member()

    # we're also running client code in tests so set dbpath as well
    # note that setting this attribute triggers connection to the run db therefore must happen after the initialization
    config.dbpath = dsn
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()
