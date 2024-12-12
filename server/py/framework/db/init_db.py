# Copyright 2023 Iguazio
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

from mlrun.common.db.sql_session import get_engine

import framework.db.sqldb.models


def init_db() -> None:
    engine = get_engine()
    partitioned_table_names = framework.db.sqldb.models.get_partitioned_table_names()
    if engine.name == "sqlite":
        tables_to_create = [
            table
            for table in framework.db.sqldb.models.Base.metadata.tables.values()
            if table.name not in partitioned_table_names
        ]
        framework.db.sqldb.models.Base.metadata.create_all(
            bind=engine, tables=tables_to_create
        )
    else:
        framework.db.sqldb.models.Base.metadata.create_all(bind=engine)
