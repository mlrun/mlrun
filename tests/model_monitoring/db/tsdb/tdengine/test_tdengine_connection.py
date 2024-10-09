import os

import pytest
import taosws

from mlrun.model_monitoring.db.tsdb.tdengine.tdengine_connection import (
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
