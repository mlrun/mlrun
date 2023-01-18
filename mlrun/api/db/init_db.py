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

from mlrun.api.db.sqldb.models import Base
from mlrun.api.db.sqldb.session import get_engine
from mlrun.config import config


def init_db(db_session: Session) -> None:
    if config.httpdb.db_type != "filedb":
        Base.metadata.create_all(bind=get_engine())
