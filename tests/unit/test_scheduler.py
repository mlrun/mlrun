# Copyright 2026 Iguazio
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

import mlrun.model


@pytest.mark.parametrize(
    "access_key, expected_secret_name",
    [
        # Normal case - secret name doesn't start with any prefix chars
        ("$ref:my-secret", "my-secret"),
        # Bug case - secret name starts with 'r' (a char in "$ref:")
        # lstrip would strip the 'r', removeprefix correctly keeps it
        ("$ref:ref-my-secret", "ref-my-secret"),
        # Bug case - secret name starts with 'e'
        ("$ref:example-secret", "example-secret"),
        # Bug case - secret name starts with 'f'
        ("$ref:function-secret", "function-secret"),
        # Bug case - secret name starts with '$'
        ("$ref:$special-secret", "$special-secret"),
        # Bug case - secret name starts with ':'
        ("$ref::colon-secret", ":colon-secret"),
        # Worst case - secret name starts with "ref" which are all chars in the prefix set
        ("$ref:reference-token", "reference-token"),
    ],
)
def test_removeprefix_strips_exact_prefix(access_key, expected_secret_name):
    """Verify that removeprefix correctly strips the $ref: prefix."""
    prefix = mlrun.model.Credentials.secret_reference_prefix
    assert access_key.startswith(prefix)

    # The correct behavior (using removeprefix)
    result = access_key.removeprefix(prefix)
    assert result == expected_secret_name


@pytest.mark.parametrize(
    "access_key, expected_secret_name",
    [
        # This case works correctly with both lstrip and removeprefix
        ("$ref:my-secret", "my-secret"),
        # These cases FAIL with lstrip but work with removeprefix
        ("$ref:ref-my-secret", "ref-my-secret"),
        ("$ref:reference-token", "reference-token"),
    ],
)
def test_lstrip_incorrectly_strips_secret_names(self, access_key, expected_secret_name):
    """
    Demonstrate that lstrip produces wrong results for secret names
    starting with characters from the prefix set.
    """
    prefix = mlrun.model.Credentials.secret_reference_prefix

    lstrip_result = access_key.lstrip(prefix)
    removeprefix_result = access_key.removeprefix(prefix)

    # removeprefix always gives the correct result
    assert removeprefix_result == expected_secret_name

    # For names starting with prefix characters, lstrip gives wrong results
    if (
        expected_secret_name.startswith(tuple(prefix))
        and expected_secret_name != "my-secret"
    ):
        # lstrip over-strips when secret name starts with chars from the set
        assert lstrip_result != expected_secret_name
        # But removeprefix gets it right
        assert removeprefix_result == expected_secret_name
