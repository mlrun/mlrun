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
#
import math

import pytest

from mlrun.data_types import ValueType
from mlrun.feature_store import Entity, FeatureSet
from mlrun.feature_store.api import delete_feature_set
from mlrun.features import Feature, Validator
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestFeatureValidator(TestMLRunSystem):
    def test_validator_types(self):
        for validator in self._validators_check_type.keys():
            for test_state, test_value in self._validators_check_type[validator]:
                feature = Feature(validator)
                feature.validator = Validator(True, "Info")

                feature.validator.set_feature(feature)
                ok, result = feature.validator.check(test_value)
                assert ok == test_state, f"Check type error, result:{result}"

    def test_feature_set_entities(self):
        # Test add entities with different syntax names, types and descriptions
        # (including featureset save)
        featureset_name = "testset_entities"
        myset = FeatureSet(featureset_name, entities=[Entity("key1")])
        myset["f1"] = Feature(ValueType.INT64, description="my f1")

        entities = [
            {"name": "key2", "vtype": ValueType.BOOL, "result": True},
            {"name": "key_3", "vtype": ValueType.INT8, "desc": "abcde", "result": True},
            {
                "name": "key__4",
                "vtype": ValueType.INT16,
                "desc": "1234",
                "result": True,
            },
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
                myset.add_entity(
                    entity["name"], entity["vtype"], entity.get("desc", None)
                )
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
        delete_feature_set(featureset_name)

    def test_feature_set_features(self):
        # Test add features with different syntax names, types and descriptions
        # (including featureset save)
        featureset_name = "testset_features"
        myset = FeatureSet(featureset_name, entities=[Entity("key")])

        features = [
            {"name": "fea1", "vtype": ValueType.BOOL, "result": True},
            {"name": "fea_2", "vtype": ValueType.INT8, "desc": "abcde", "result": True},
            {
                "name": "fea__3",
                "vtype": ValueType.INT16,
                "desc": "1234",
                "result": True,
            },
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
        delete_feature_set(featureset_name)

    _validators_check_type = {
        # ValueType: [(expected result, testing value), ...]
        # BOOL is without validation
        ValueType.BOOL: [(True, True), (True, False), (True, "test"), (True, 60)],
        ValueType.INT8: [(True, -128), (True, 127), (False, -129), (False, 128)],
        ValueType.INT16: [
            (True, -32768),
            (True, 32767),
            (False, -32769),
            (False, 32768),
        ],
        ValueType.INT32: [
            (True, -2147483648),
            (True, 2147483647),
            (False, -2147483649),
            (False, 2147483648),
        ],
        ValueType.INT64: [
            (True, -9223372036854775808),
            (True, 9223372036854775807),
            (False, -9223372036854775809),
            (False, 9223372036854775808),
        ],
        ValueType.INT128: [
            (True, -math.pow(2, 127)),
            (True, math.pow(2, 127) - 1),
            (False, -math.pow(2, 128)),
            (False, math.pow(2, 128)),
        ],
        ValueType.UINT8: [(True, 0), (True, 255), (False, 256), (False, -1)],
        ValueType.UINT16: [(True, 0), (True, 65535), (False, -1), (False, 65536)],
        ValueType.UINT32: [
            (True, 0),
            (True, 4294967295),
            (False, -1),
            (False, 4294967296),
        ],
        ValueType.UINT64: [
            (True, 0),
            (True, 18446744073709551615),
            (False, -1),
            (False, 18446744073709551616),
        ],
        ValueType.UINT128: [
            (True, 0),
            (True, math.pow(2, 128)),
            (False, -1),
            (False, math.pow(2, 129)),
        ],
        ValueType.FLOAT: [
            (True, 3.1415),
            (True, -273.15),
            (False, "WDC"),
            (True, "6378"),
            (False, ""),
        ],
        ValueType.DOUBLE: [
            (True, 11.2),
            (True, -459.67),
            (False, "WDC"),
            (True, "6378"),
            (False, ""),
        ],
        # STRING is without validation (it can be use later for compatibility test between Code pages, etc.)
        ValueType.STRING: [
            (True, "aa"),
            (True, "\x00\x01"),
            (True, "\x00\x01\x02\x03\xff"),
            (True, False),
            (True, 3.1415927),
        ],
    }
