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
from datetime import datetime, timedelta, timezone

import pytest

import mlrun
from mlrun.data_types.data_types import ValueType
from mlrun.datastore import ParquetSource
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
    "pass_run_id, is_single_file, target_path, expected_path",
    [
        (True, False, "v3io:///bigdata/{run_id}", "v3io:///bigdata/run_id_val/"),
        (
            True,
            True,
            "v3io:///bigdata/{run_id}/file.parquet",
            "v3io:///bigdata/run_id_val/file.parquet",
        ),
        (False, False, "v3io:///bigdata/", "v3io:///bigdata/"),
        (False, True, "v3io:///bigdata/file.parquet", "v3io:///bigdata/file.parquet"),
        (True, False, "v3io:///bigdata/", "v3io:///bigdata/run_id_val/"),
        (
            True,
            True,
            "v3io:///bigdata/file.parquet",
            "v3io:///bigdata/run_id_val/file.parquet",
        ),
        (False, False, "v3io:///bigdata/{run_id}", None),
        (False, True, "v3io:///bigdata/{run_id}/file.parquet", None),
    ],
)
def test_different_target_path_scenarios_for_run_id(
    pass_run_id, is_single_file, target_path, expected_path
):
    run_id = "run_id_val" if pass_run_id else None

    if not expected_path:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            TargetPathObject(target_path, run_id, is_single_file)
    else:
        tp_obj = TargetPathObject(target_path, run_id, is_single_file)
        assert (
            mlrun.model.RUN_ID_PLACE_HOLDER in tp_obj.get_templated_path()
        ) == pass_run_id
        assert mlrun.model.RUN_ID_PLACE_HOLDER not in tp_obj.get_absolute_path()
        assert tp_obj.get_absolute_path() == expected_path


@pytest.mark.parametrize(
    "time_for_source, is_through_init, should_succeed, time_delta",
    [
        (None, False, True, None),
        ("", False, True, None),
        (datetime(2021, 5, 25, 10, 30, 29, 592000), False, True, None),
        (datetime(2021, 5, 25, 10, 30, 29, 592000), True, True, None),
        (datetime(2021, 5, 25, 10, 30, 29, 592000, timezone.utc), False, True, 0),
        (datetime(2021, 5, 25, 10, 30, 29, 592000, timezone.utc), True, True, 0),
        ("2021-05-25T10:30:29.abc", False, False, None),
        ("2021-05-25T10:30:29.abc", True, False, None),
        ("2021-05-25T10:30:29.592", False, True, None),
        ("2021-05-25T10:30:29.592", True, True, None),
        ("2021-05-25T10:30:29.592Z", False, True, 0),
        ("2021-05-25T10:30:29.592Z", True, True, 0),
        ("2021-05-25T10:30:29.592+00:00", False, True, 0),
        ("2021-05-25T10:30:29.592+00:00", True, True, 0),
        ("2021-05-25T10:30:29.592+02:00", False, True, timedelta(hours=2)),
        ("2021-05-25T10:30:29.592+02:00", True, True, timedelta(hours=2)),
    ],
)
def test_parquet_source_with_iso_start_or_end_time(
    time_for_source, is_through_init, should_succeed, time_delta
):
    def _test_parquet_source_with_iso_start_or_end_time(
        time_for_source, is_through_init, time_delta
    ):
        if time_delta is None:
            tzinfo = None
        else:
            tzinfo = timezone.utc if time_delta == 0 else timezone(time_delta)
        actual = datetime(2021, 5, 25, 10, 30, 29, 592000, tzinfo)
        source = ParquetSource(
            "srcpar",
            path="tmp/doesnt/matter",
            start_time=time_for_source if is_through_init else None,
            end_time=time_for_source if is_through_init else None,
        )
        if not is_through_init:
            source.start_time = time_for_source
            source.end_time = time_for_source

        if source.start_time:
            assert source.start_time == actual
        if source.end_time:
            assert source.end_time == actual

    if should_succeed:
        _test_parquet_source_with_iso_start_or_end_time(
            time_for_source, is_through_init, time_delta
        )
    else:
        with pytest.raises(ValueError, match=r".*Invalid isoformat string:.*"):
            _test_parquet_source_with_iso_start_or_end_time(
                time_for_source, is_through_init, time_delta
            )
