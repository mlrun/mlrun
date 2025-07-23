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

import mlrun

import framework.utils.db.utils


@pytest.mark.integration
def test_set_mysql_modes(
    db_util: framework.utils.db.utils.DBUtil,
):
    original = db_util.get_current_configurations()
    if "PIPES_AS_CONCAT" in original:
        raise AssertionError(
            "The test is not applicable, 'PIPES_AS_CONCAT' is already set."
        )
    raw_configs = mlrun.mlconf.httpdb.db.mysql.modes.split(",") + [
        "PIPES_AS_CONCAT",
        "ONLY_FULL_GROUP_BY",  # This is a default setting.
    ]
    try:
        db_util.set_configurations(raw_configs)
        updated = set(db_util.get_current_configurations())
        original["PIPES_AS_CONCAT"] = True
        assert set(original) == set(updated)
    finally:
        db_util.set_configurations(original)

    restored = db_util.get_current_configurations()
    assert restored == original


@pytest.mark.integration
@pytest.mark.parametrize("noop_key", ["nil", "none"])
def test_set_configurations_noop_values_are_ignored(
    db_util: framework.utils.db.utils.DBUtil,
    noop_key: str,
):
    original = list(db_util.get_current_configurations())

    db_util.set_configurations([noop_key])
    after = list(db_util.get_current_configurations())

    assert (
        after == original
    ), f"Configuration changed after setting noop value '{noop_key}'"
