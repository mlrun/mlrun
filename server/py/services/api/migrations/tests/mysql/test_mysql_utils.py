# Copyright 2025 Iguazio
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
from pytest_mock_resources import MysqlConfig

from framework.utils.db.utils import DBUtil


def _current_sql_mode(util: DBUtil) -> str:
    conn = util._get_driver().connect(**util._connection_kwargs())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT @@GLOBAL.sql_mode;")
            return (cur.fetchone()[0] or "").strip()
    finally:
        conn.close()

@pytest.mark.integration
def test_mysql_apply_strict_all_tables_live(pmr_mysql_container: MysqlConfig):
    util = DBUtil()
    print(type(pmr_mysql_container))

    original = _current_sql_mode(util)
    current_modes = {m.strip() for m in original.split(",") if m.strip()}
    if "STRICT_ALL_TABLES" in current_modes:
        raise AssertionError(
            "The test is not applicable, 'STRICT_ALL_TABLES' is already set."
        )

    # apply
    util.set_modes("STRICT_ALL_TABLES")
    assert _current_sql_mode(util) == "STRICT_ALL_TABLES"

    # restore
    util.set_modes(original)
    assert _current_sql_mode(util) == original
