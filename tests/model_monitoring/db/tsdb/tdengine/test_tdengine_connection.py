# Copyright 2024 Iguazio
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

import os

import pytest
import taosws
from taoswswrap.tdengine_connection import (
    Field,
    QueryResult,
    TDEngineConnection,
)

connection_string = os.getenv("MLRUN_MODEL_ENDPOINT_MONITORING__TSDB_CONNECTION")


def is_tdengine_defined() -> bool:
    return connection_string is not None and connection_string.startswith("taosws://")


@pytest.mark.skipif(not is_tdengine_defined(), reason="TDEngine is not defined")
def test_tdengine_connection():
    conn = TDEngineConnection(connection_string)

    res = conn.run(
        statements=[
            "DROP DATABASE IF EXISTS mydb",
            "CREATE DATABASE mydb",
            "USE mydb",
            "CREATE STABLE mystable (column1 TIMESTAMP, column2 FLOAT) TAGS (tag1 INT);",
            "CREATE TABLE mytable USING mystable TAGS (1)",
            "INSERT INTO mytable VALUES (1728444786455, 1)",
        ],
        query="SELECT * FROM mytable",
    )
    assert res == QueryResult(
        [("2024-10-09 11:33:06.455 +08:00", 1.0)],
        [Field("column1", "TIMESTAMP", 8), Field("column2", "FLOAT", 4)],
    )


@pytest.mark.skipif(not is_tdengine_defined(), reason="TDEngine is not defined")
def test_tdengine_connection_error_propagation():
    conn = TDEngineConnection(connection_string)

    with pytest.raises(taosws.QueryError, match="Internal error: `Database not exist`"):
        conn.run(statements="USE idontexist")


@pytest.mark.skipif(not is_tdengine_defined(), reason="TDEngine is not defined")
def test_tdengine_connection_create_db():
    conn = TDEngineConnection(connection_string)

    res = conn.run(statements="CREATE DATABASE IF NOT EXISTS mydb")
    assert res is None
