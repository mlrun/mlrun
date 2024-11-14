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

from functools import partial
from typing import Optional, TypeVar, Union

from .mysql import ModelEndpointsTable as MySQLModelEndpointsTable
from .sqlite import ModelEndpointsTable as SQLiteModelEndpointsTable

MySQLTableType = TypeVar("MySQLTableType")
SQLiteTableType = TypeVar("SQLiteTableType")

_MYSQL_SCHEME = "mysql:"


def _get_sql_table(
    *,
    mysql_table: MySQLTableType,
    sqlite_table: SQLiteTableType,
    connection_string: Optional[str] = None,
) -> Union[MySQLTableType, SQLiteTableType]:
    """
    Return a SQLAlchemy table for MySQL or SQLite according to the connection string.
    Note: this function should not be directly used in other modules.
    """
    if connection_string and _MYSQL_SCHEME in connection_string:
        return mysql_table
    return sqlite_table


_get_model_endpoints_table = partial(
    _get_sql_table,
    mysql_table=MySQLModelEndpointsTable,
    sqlite_table=SQLiteModelEndpointsTable,
)
