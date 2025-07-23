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
import unittest.mock

import pytest

import framework.utils.db.utils


@pytest.mark.parametrize("empty_key", ["nil", "none"])
def test_set_configurations_skips_when_nil_or_none(
    db_util: framework.utils.db.utils.DBUtil,
    monkeypatch,
    empty_key: str,
):
    mocked_apply = unittest.mock.MagicMock()
    monkeypatch.setattr(db_util, "_apply_configurations", mocked_apply)

    db_util.set_configurations([empty_key])

    mocked_apply.assert_not_called()


def test_set_configurations_skips_when_none(
    db_util: framework.utils.db.utils.DBUtil,
    monkeypatch,
):
    mocked_apply = unittest.mock.MagicMock()
    monkeypatch.setattr(db_util, "_apply_configurations", mocked_apply)

    db_util.set_configurations(None)
    mocked_apply.assert_not_called()


@pytest.mark.parametrize("item", ["STRICT_TRANS_TABLES", "NO_ZERO_IN_DATE"])
def test_set_configurations_called_with_modes(
    db_util: framework.utils.db.utils.DBUtil,
    monkeypatch,
    item: str,
):
    mocked_apply = unittest.mock.MagicMock()
    connection_mock = unittest.mock.MagicMock()
    monkeypatch.setattr(db_util, "_apply_configurations", mocked_apply)
    monkeypatch.setattr(db_util, "_get_connection", connection_mock)

    db_util.set_configurations([item])
    mocked_apply.assert_called_with(unittest.mock.ANY, [item])
