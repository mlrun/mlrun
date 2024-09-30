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


import pytest

import server.api.utils.db.mysql


@pytest.mark.parametrize(
    "http_dsn, expected_output",
    [
        (
            "mysql+pymysql://root:pass@localhost:3307/mlrun",
            {
                "username": "root",
                "password": "pass",
                "host": "localhost",
                "port": "3307",
                "database": "mlrun",
            },
        ),
        (
            "mysql+pymysql://root@192.168.228.104:3306/mlrun",
            {
                "username": "root",
                "password": None,
                "host": "192.168.228.104",
                "port": "3306",
                "database": "mlrun",
            },
        ),
        ("mysql+pymysql://@localhost:3307/mlrun", None),
        ("mysql+pymysql://root:pass@localhost:3307", None),
        ("sqlite:///db/mlrun.db?check_same_thread=false", None),
    ],
)
def test_get_mysql_dsn_data(
    http_dsn: str, expected_output: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MLRUN_HTTPDB__DSN", http_dsn)
    dns_data = server.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data()
    assert dns_data == expected_output
