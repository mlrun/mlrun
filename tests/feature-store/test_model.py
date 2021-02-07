# Copyright 2018 Iguazio
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

from mlrun.data_types import ValueType
from mlrun.feature_store.common import parse_feature_string
from mlrun.feature_store import FeatureSet, Feature, Entity


def test_feature_set():
    myset = FeatureSet("set1", entities=[Entity("key")])
    myset["f1"] = Feature(ValueType.INT64, description="my f1")

    assert list(myset.spec.entities.keys()) == ["key"], "index wasnt set"
    assert list(myset.spec.features.keys()) == ["f1"], "feature wasnt set"


def test_features_parser():
    cases = [
        {"feature": "set1", "result": None, "error": True},
        {"feature": "set1 as x", "result": None, "error": True},
        {"feature": "set1.f1", "result": ("set1", "f1", None)},
        {"feature": "proj/set1.f1", "result": ("proj/set1", "f1", None)},
        {"feature": "set1. f1", "result": ("set1", "f1", None)},
        {"feature": "set1 . f1 ", "result": ("set1", "f1", None)},
        {"feature": "set1.*", "result": ("set1", "*", None)},
        {"feature": "set1.f2 as x", "result": ("set1", "f2", "x")},
    ]
    for case in cases:
        try:
            result = parse_feature_string(case["feature"])
        except Exception as e:
            assert case.get("error", False), f"got unexpected error {e}"
            continue
        assert result == case["result"]
