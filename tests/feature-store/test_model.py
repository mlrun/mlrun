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

from mlrun.data_types.data_types import ValueType
from mlrun.feature_store import Entity, Feature, FeatureSet
from mlrun.feature_store.common import parse_feature_string


def test_feature_set():
    myset = FeatureSet("set1", entities=[Entity("key")])
    myset["f1"] = Feature(ValueType.INT64, description="my f1")

    assert list(myset.spec.entities.keys()) == ["key"], "index wasnt set"
    assert list(myset.spec.features.keys()) == ["f1"], "feature wasnt set"


def test_feature_set_entities():
    # Test add entities with different syntax names, types and descriptions
    # (including featureset save)
    myset = FeatureSet("testset_entities", entities=[Entity("key1")])
    myset["f1"] = Feature(ValueType.INT64, description="my f1")

    entities = [
        {"name": "key2", "vtype": ValueType.BOOL, "result": True},
        {"name": "key_3", "vtype": ValueType.INT8, "desc": "abcde", "result": True},
        {"name": "key__4", "vtype": ValueType.INT16, "desc": "1234", "result": True},
        {
            "name": "key 5",
            "vtype": ValueType.INT32,
            "desc": "+ěščřžýáíéúů",
            "result": True,
        },
        {"name": "key_6", "vtype": ValueType.INT64, "result": True},
        {"name": "key_7", "vtype": ValueType.INT128, "result": True},
        {"name": "key_8", "vtype": ValueType.UINT8, "result": True},
        {"name": "key__9", "vtype": ValueType.UINT16, "result": True},
        {"name": "key 10", "vtype": ValueType.UINT32, "result": True},
        {"name": "key_11", "vtype": ValueType.UINT64, "result": True},
        {"name": "key_12", "vtype": ValueType.UINT128, "result": True},
        {"name": "key_13", "vtype": ValueType.FLOAT, "result": True},
        {"name": "key_14", "vtype": ValueType.FLOAT16, "result": True},
        {"name": "key_15", "vtype": ValueType.DOUBLE, "result": True},
        {"name": "key_16", "vtype": ValueType.BFLOAT16, "result": True},
        {"name": "key_17", "vtype": ValueType.BYTES, "result": True},
        {"name": "key_18", "vtype": ValueType.STRING, "result": True},
        {"name": "key_19", "vtype": ValueType.DATETIME, "result": True},
    ]

    for entity in entities:
        try:
            myset.add_entity(entity["name"], entity["vtype"], entity.get("desc", None))
            myset.save()
        except Exception as exc:
            assert entity.get("result", False) is False, (
                f"got unexpected error "
                f"for for entity '{entity.get('name', None)}'"
                f"with type '{entity.get('vtype', None)}', "
                f"error '{exc}''"
            )
            continue
        assert entity["result"], (
            f"unexpected value for entity "
            f"'{entity.get('name', None)}' with "
            f"type '{entity.get('vtype', None)}'"
        )


def test_feature_set_features():
    # Test add features with different syntax names, types and descriptions
    # (including featureset save)
    myset = FeatureSet("testset_features", entities=[Entity("key")])

    features = [
        {"name": "fea1", "vtype": ValueType.BOOL, "result": True},
        {"name": "fea_2", "vtype": ValueType.INT8, "desc": "abcde", "result": True},
        {"name": "fea__3", "vtype": ValueType.INT16, "desc": "1234", "result": True},
        {
            "name": "fea 4",
            "vtype": ValueType.INT32,
            "desc": "+ěščřžýáíéúů",
            "result": True,
        },
        {"name": "fea_5", "vtype": ValueType.INT64, "result": True},
        {"name": "fea_6", "vtype": ValueType.INT128, "result": True},
        {"name": "fea_7", "vtype": ValueType.UINT8, "result": True},
        {"name": "fea__8", "vtype": ValueType.UINT16, "result": True},
        {"name": "fea 9", "vtype": ValueType.UINT32, "result": True},
        {"name": "fea_10", "vtype": ValueType.UINT64, "result": True},
        {"name": "fea_11", "vtype": ValueType.UINT128, "result": True},
        {"name": "fea_12", "vtype": ValueType.FLOAT, "result": True},
        {"name": "fea_13", "vtype": ValueType.FLOAT16, "result": True},
        {"name": "fea_14", "vtype": ValueType.DOUBLE, "result": True},
        {"name": "fea_15", "vtype": ValueType.BFLOAT16, "result": True},
        {"name": "fea_16", "vtype": ValueType.BYTES, "result": True},
        {"name": "fea_17", "vtype": ValueType.STRING, "result": True},
        {"name": "fea_18", "vtype": ValueType.DATETIME, "result": True},
    ]

    for feature in features:
        try:
            myset.add_feature(
                Feature(
                    value_type=feature["vtype"],
                    description=feature.get("desc", None),
                    name=feature["name"],
                )
            )
            myset.save()
        except Exception as exc:
            assert feature.get("result", False) is False, (
                f"got unexpected error "
                f"for for entity '{feature.get('name', None)}'"
                f"with type '{feature.get('vtype', None)}', "
                f"error '{exc}''"
            )
            continue
        assert feature["result"], (
            f"unexpected value for entity "
            f"'{feature.get('name', None)}' with "
            f"type '{feature.get('vtype', None)}'"
        )


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
        except Exception as exc:
            assert case.get("error", False), f"got unexpected error {exc}"
            continue
        assert result == case["result"]
