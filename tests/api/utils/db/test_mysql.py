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

import os

import pytest

import server.api.utils.db.mysql


@pytest.mark.parametrize(
    "http_dns",
    [
        "mysql+pymysql://root:pass@localhost:3307/mlrun",
        "mysql+pymysql://root@localhost:3307/mlrun",
    ],
)
def test_get_mysql_dsn_data(http_dns):
    os.environ["MLRUN_HTTPDB__DSN"] = http_dns
    dns_data = server.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data()
    assert dns_data is not None
