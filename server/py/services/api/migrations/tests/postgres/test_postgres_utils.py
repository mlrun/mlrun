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
from framework.utils.db.utils import DBUtil


def test_postgres_apply_modes_live(pmr_postgres_container, patched_dsn):
    util = DBUtil()  # -> UtilPostgres

    def _show(name):
        conn = util._get_driver().connect(**util._connection_kwargs())
        try:
            with conn.cursor() as cur:
                cur.execute(f"SHOW {name};")
                return cur.fetchone()[0].strip()
        finally:
            conn.close()

    guc = "work_mem"
    original = _show(guc)
    new_val = "64MB" if original != "64MB" else "32MB"

    util.set_modes(f"{guc}={new_val}")
    assert _show(guc) == new_val

    util.set_modes(f"{guc}={original}")
    assert _show(guc) == original
