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
def test_postgres_apply_modes_live(
    db_util: framework.utils.db.utils.DBUtil,
):
    configs = db_util.get_current_configurations()
    old_value = configs.get("work_mem")

    assert (
        old_value != "65536"
    ), "The test is not applicable, 'work_mem' is already set to '65536'."

    # apply new setting
    db_util.set_configurations({"work_mem": 65536})

    assert db_util.get_current_configurations()["work_mem"] == "65536"

    # restore original
    db_util.set_configurations({"work_mem": old_value})

    assert db_util.get_current_configurations()["work_mem"] == old_value

    # sanity: ensure only work_mem changed back
    final = db_util.get_current_configurations()

    assert final.get("work_mem") == old_value
