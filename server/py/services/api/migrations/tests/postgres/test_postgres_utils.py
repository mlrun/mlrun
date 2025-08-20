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

import framework.utils.db.utils

pytest.importorskip(
    "psycopg2",
    reason="psycopg2 not installed",
)


@pytest.mark.integration
def test_postgres_apply_work_mem_live(
    db_util: framework.utils.db.utils.DBUtil,
):
    original = db_util.get_current_configurations()
    old_value = original.get("work_mem")

    assert old_value != "65536", "Test requires 'work_mem' to not be '65536' initially"

    try:
        db_util.set_configurations({"work_mem": 65536})
        updated = db_util.get_current_configurations()
        assert updated.get("work_mem") == "65536", "'work_mem' was not updated"
    finally:
        db_util.set_configurations({"work_mem": old_value})
    restored = db_util.get_current_configurations()
    assert restored.get("work_mem") == old_value, "'work_mem' was not restored"


@pytest.mark.integration
@pytest.mark.parametrize("noop_key", ["nil", "none"])
def test_postgres_set_configurations_noop_values_ignored(
    db_util: framework.utils.db.utils.DBUtil,
    noop_key: str,
):
    original = dict(db_util.get_current_configurations())

    db_util.set_configurations({noop_key: "some_val"})
    after = dict(db_util.get_current_configurations())

    assert after == original, f"Config changed after noop key '{noop_key}'"
