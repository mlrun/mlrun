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
from typing import Optional

import pytest

import framework.utils.db.utils


@pytest.mark.parametrize(
    "http_dsn, expected_output",
    [
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
    ],
)
def test_get_dsn_data(
    http_dsn: str,
    expected_output: Optional[dict],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLRUN_HTTPDB__DSN", http_dsn)
    parsed = framework.utils.db.utils.DBUtil.get_parsed_dsn()

    if expected_output is None:
        assert not parsed.is_valid()
    else:
        for field, expected in expected_output.items():
            assert getattr(parsed, field) == expected
