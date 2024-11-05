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

import pytest

import services.api.utils.singletons.db


@pytest.mark.parametrize(
    "dsn,expected_masked_dsn",
    [
        (
            "mysql+pymysql://root:pass@localhost:3307/mlrun",
            "mysql+pymysql://***:***@localhost:3307/mlrun",
        ),
        (
            "mysql+pymysql://root@localhost:3307/mlrun",
            "mysql+pymysql://***@localhost:3307/mlrun",
        ),
        (
            "sqlite:///db/mlrun.db?check_same_thread=false",
            "sqlite:///db/mlrun.db?check_same_thread=false",
        ),
    ],
)
def test_masked_dsn(dsn, expected_masked_dsn):
    masked_dsn = services.api.utils.singletons.db._mask_dsn(dsn)
    assert masked_dsn == expected_masked_dsn
