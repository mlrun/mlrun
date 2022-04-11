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
import mlrun
import pytest
from mlrun.data_types.data_types import ValueType
from mlrun.feature_store import Entity, Feature, FeatureSet
from mlrun.feature_store.common import parse_feature_string

from mlrun.model import TargetPathObject


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
        except Exception as exc:
            assert case.get("error", False), f"got unexpected error {exc}"
            continue
        assert result == case["result"]


@pytest.mark.parametrize(
    "raise_error, pass_run_id, is_single_file, target_path",
    [
        (False, True, False, "v3io:///bigdata/{run_id}"),
        (False, True, True, "v3io:///bigdata/{run_id}/file.parquet"),
        (False, False, False, "v3io:///bigdata/"),
        (False, False, True, "v3io:///bigdata/file.parquet"),
        (False, True, False, "v3io:///bigdata/"),
        (False, True, True, "v3io:///bigdata/file.parquet"),
        (True, False, False, "v3io:///bigdata/{run_id}"),
        (True, False, True, "v3io:///bigdata/{run_id}/file.parquet"),
    ],
)
def test_different_target_path_scenarios_for_run_id(
    raise_error, pass_run_id, is_single_file, target_path
):
    run_id = "run_id_val" if pass_run_id else None

    if raise_error:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            TargetPathObject(target_path, run_id, is_single_file)
    else:
        tp_obj = TargetPathObject(target_path, run_id, is_single_file)
        assert (mlrun.model.RUN_ID_PLACE_HOLDER in tp_obj.get_templated_path()) == (
            (pass_run_id and not is_single_file)
            or mlrun.model.RUN_ID_PLACE_HOLDER in target_path
        )
        assert mlrun.model.RUN_ID_PLACE_HOLDER not in tp_obj.get_absolute_path()