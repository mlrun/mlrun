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

from mlrun.feature_store.api import norm_column_name
from mlrun.feature_store.common import parse_feature_string


# ML-7453
def test_parse_feature_string_with_dot_in_feature_set_name():
    feature_set, feature, alias = parse_feature_string(
        "monitoring-llm-server-Qwen-Qwen2-0.5B-latest.*"
    )
    assert feature_set == "monitoring-llm-server-Qwen-Qwen2-0.5B-latest"
    assert feature == "*"
    assert alias is None


def test_parse_feature_string_with_alias():
    feature_set, feature, alias = parse_feature_string("fset.feature as alias")
    assert feature_set == "fset"
    assert feature == "feature"
    assert alias == "alias"


# ML-12672
@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("feat 1", "feat_1"),
        ("b (C)", "b_C"),
        ("Last   for df ", "Last___for_df_"),
        ("class (0-4) ", "class_0-4_"),
        ("already_ok", "already_ok"),
    ],
)
def test_norm_column_name_special_chars(raw_name, expected):
    assert norm_column_name(raw_name) == expected
