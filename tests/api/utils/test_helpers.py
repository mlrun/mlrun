# Copyright 2024 Iguazio
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

import server.api.utils.helpers


@pytest.mark.parametrize(
    "client_version, min_versions, expected_compatible",
    [
        ("1.7.0", ["1.7.0"], True),
        ("1.7.0-rc1", ["1.7.0"], False),
        ("1.7.0", ["1.8.0"], False),
        ("1.7.0-rc1", ["1.8.0"], False),
        ("1.7.1", ["1.8.0", "1.7.1"], True),
        ("1.7.1-rc1", ["1.8.0", "1.7.1"], False),
        ("1.8.0", [], False),
        ("1.8.0", ["1.8.0"], True),
        ("1.8.0-rc13", ["1.8.0-rc12"], True),
        ("1.8.0-rc13", ["1.8.0-rc13"], True),
        ("1.8.0-rc13", ["1.8.0-rc14"], False),
        ("1.8.0", ["1.8.1"], False),
        ("1.9.0", ["1.8.1"], True),
        ("0.0.0-unstable", [], True),
        ("0.0.0-unstable", ["1.9.0"], True),
    ],
)
def test_validate_client_version(client_version, min_versions, expected_compatible):
    assert (
        server.api.utils.helpers.validate_client_version(client_version, *min_versions)
        == expected_compatible
    )
