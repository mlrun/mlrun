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
from sqlalchemy.orm import Session

from mlrun.api.db.sqldb.session import create_session as sqldb_create_session
from mlrun.config import config


def create_session(db_type=None) -> Session:
    db_type = db_type or config.httpdb.db_type
    if db_type == "filedb":
        return None
    else:
        return sqldb_create_session()


def close_session(db_session):

    # will be None when it's filedb session
    if db_session is not None:
        db_session.close()


def run_function_with_new_db_session(func):
    session = create_session()
    try:
        result = func(session)
        return result
    finally:
        close_session(session)
