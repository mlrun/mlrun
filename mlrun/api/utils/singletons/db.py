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
from mlrun.api.db.base import DBInterface
from mlrun.api.db.filedb.db import FileDB
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import create_session
from mlrun.config import config
from mlrun.utils import logger

# TODO: something nicer
db: DBInterface = None


def get_db() -> DBInterface:
    global db
    return db


def initialize_db(override_db=None):
    global db
    if override_db:
        db = override_db
        return
    if config.httpdb.db_type == "filedb":
        logger.info("Creating file db")
        db = FileDB(config.httpdb.dirpath)
        db.initialize(None)
    else:
        logger.info("Creating sql db")
        db = SQLDB(config.httpdb.dsn)
        db_session = None
        try:
            db_session = create_session()
            db.initialize(db_session)
        finally:
            db_session.close()
