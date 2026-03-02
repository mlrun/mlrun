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

import pytest

import framework.utils.db.dsn


@pytest.mark.parametrize(
    "dsn, expected_output",
    [
        (
            "mysql+pymysql://test_user1:p~Xfi70|-ZM#U~Rf_5Ht2N<:ha9?@mlrun-test-db-instance.cucgqhjk51rp.us-east-2.rds.amazonaws.com:3306/mlrundb_vmdev214.lab.iguazeng.com",
            {
                "username": "test_user1",
                "password": "p~Xfi70|-ZM#U~Rf_5Ht2N<:ha9?",
                "host": "mlrun-test-db-instance.cucgqhjk51rp.us-east-2.rds.amazonaws.com",
                "port": 3306,
                "database": "mlrundb_vmdev214.lab.iguazeng.com",
            },
        ),
        (
            "mysql+pymysql://root:pass@localhost:3307/mlrun",
            {
                "username": "root",
                "password": "pass",
                "host": "localhost",
                "port": 3307,
                "database": "mlrun",
            },
        ),
        (
            "mysql+pymysql://root:pass@@localhost:3307/mlrun",
            {
                "username": "root",
                "password": "pass@",
                "host": "localhost",
                "port": 3307,
                "database": "mlrun",
            },
        ),
        (
            "mysql+pymysql://root@192.168.228.104:3306/mlrun",
            {
                "username": "root",
                "password": None,
                "host": "192.168.228.104",
                "port": 3306,
                "database": "mlrun",
            },
        ),
        ("mysql+pymysql://@localhost:3307/mlrun", None),
        ("mysql+pymysql://root:pass@localhost:3307", None),
        (
            "sqlite:///db/mlrun.db?check_same_thread=false",
            {
                "username": None,
                "password": None,
                "host": None,
                "port": None,
                "database": None,
            },
        ),
        (
            "sqlite://",
            {
                "username": None,
                "password": None,
                "host": None,
                "port": None,
                "database": None,
            },
        ),
        (
            "sqlite:///:memory:",
            {
                "username": None,
                "password": None,
                "host": None,
                "port": None,
                "database": None,
            },
        ),
        (
            "sqlite:////absolute/path/to/my.db",
            {
                "username": None,
                "password": None,
                "host": None,
                "port": None,
                "database": None,
            },
        ),
        (
            "mysql+pymysql://root:pw@db_host:3306/mlrun",
            {
                "username": "root",
                "password": "pw",
                "host": "db_host",
                "port": 3306,
                "database": "mlrun",
            },
        ),
        (
            "mysql://root:pw@localhost:3306/mlrun",
            {
                "username": "root",
                "password": "pw",
                "host": "localhost",
                "port": 3306,
                "database": "mlrun",
            },
        ),
        ("mysql+pymysql://root:pw@localhost:70000/mlrun", None),
        ("oracle://root:pw@localhost:1521/xe", None),
        ("mysql+pymysql://root:pw@:3306/mlrun", None),
        # Encoded '@' in password
        (
            "mysql+pymysql://user:p%40ssw%40rd@db-host:3306/mlrun",
            {
                "username": "user",
                "password": "p@ssw@rd",
                "host": "db-host",
                "port": 3306,
                "database": "mlrun",
            },
        ),
        # Password present, username missing → invalid
        ("mysql+pymysql://:pw@localhost:3306/mlrun", None),
        # No credentials & no '@' → invalid
        ("mysql+pymysql://localhost:3306/mlrun", None),
        # Non‑numeric port text → invalid
        ("mysql+pymysql://root:pw@localhost:abC/mlrun", None),
        # Missing port entirely → invalid
        ("mysql+pymysql://root:pw@localhost/mlrun", None),
        # MySQL with query params (valid, configs stored separately)
        (
            "mysql+pymysql://root:pw@db:3306/mlrun?charset=utf8mb4&ssl=true",
            {
                "username": "root",
                "password": "pw",
                "host": "db",
                "port": 3306,
                "database": "mlrun",
            },
        ),
        # Postgres dialect
        (
            "postgresql://alice:pw@pg-host:5432/analytics",
            {
                "username": "alice",
                "password": "pw",
                "host": "pg-host",
                "port": 5432,
                "database": "analytics",
            },
        ),
        # SQLite path traversal attempt is invalid
        ("sqlite:///../etc/passwd", None),
        # SQLite file with query string (valid)
        (
            "sqlite:///var/data/app.db?mode=ro&cache=shared",
            {
                "username": None,
                "password": None,
                "host": None,
                "port": None,
                "database": None,
            },
        ),
        # password contains both %40 (encoded) and raw '@'
        ("mysql+pymysql://u:foo%40ba@r@host:3306/db", None),  # should be invalid
    ],
)
def test_parse_dsn(
    dsn: str,
    expected_output: dict | None,
    monkeypatch: pytest.MonkeyPatch,
):
    parsed = framework.utils.db.dsn.Dsn(dsn)

    if expected_output is None:
        assert not parsed.is_valid()
    else:
        for field, expected in expected_output.items():
            assert getattr(parsed, field) == expected
