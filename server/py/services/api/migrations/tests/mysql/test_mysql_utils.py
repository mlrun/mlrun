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


@pytest.mark.integration
def test_mysql_apply_strict_all_tables_live(pmr_mysql_container: MysqlConfig):
    util = DBUtil()
    print(type(pmr_mysql_container))

    original = list(util.get_current_configurations())
    if "PIPES_AS_CONCAT" in original:
        raise AssertionError(
            "The test is not applicable, 'PIPES_AS_CONCAT' is already set."
        )

    # apply
    util.set_configurations(["PIPES_AS_CONCAT"])
    assert util.get_current_configurations()

    # restore
    util.set_configurations(original)
    assert list(util.get_current_configurations()) == original
