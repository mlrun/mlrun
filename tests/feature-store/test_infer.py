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
import unittest.mock

import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fs
from mlrun.data_types import InferOptions
from mlrun.datastore.targets import ParquetTarget
from mlrun.feature_store import Entity
from mlrun.feature_store.api import _infer_from_static_df
from tests.conftest import tests_root_directory

this_dir = f"{tests_root_directory}/feature-store/"

expected_schema = [
    {"name": "bad", "value_type": "int"},
    {"name": "department", "value_type": "str"},
    {"name": "room", "value_type": "int"},
    {"name": "hr", "value_type": "float"},
    {"name": "hr_is_error", "value_type": "bool"},
    {"name": "rr", "value_type": "int"},
    {"name": "rr_is_error", "value_type": "bool"},
    {"name": "spo2", "value_type": "int"},
    {"name": "spo2_is_error", "value_type": "bool"},
    {"name": "movements", "value_type": "float"},
    {"name": "movements_is_error", "value_type": "bool"},
    {"name": "turn_count", "value_type": "float"},
    {"name": "turn_count_is_error", "value_type": "bool"},
    {"name": "is_in_bed", "value_type": "int"},
    {"name": "is_in_bed_is_error", "value_type": "bool"},
    {"name": "timestamp", "value_type": "str"},
]


def test_infer_from_df():
    key = "patient_id"
    df = pd.read_csv(this_dir + "testdata.csv")
    df.set_index(key, inplace=True)
    featureset = fs.FeatureSet("testdata")
    _infer_from_static_df(df, featureset, options=InferOptions.all())
    # print(featureset.to_yaml())

    # test entity infer
    assert len(featureset.spec.entities) == 1, "entity not properly inferred"
    assert (
        list(featureset.spec.entities.keys())[0] == key
    ), "entity key not properly inferred"
    assert (
        list(featureset.spec.entities.values())[0].value_type == "str"
    ), "entity type not properly inferred"

    # test infer features
    assert (
        featureset.spec.features.to_dict() == expected_schema
    ), "did not infer schema properly"

    preview = featureset.status.preview
    # by default preview should be 20 lines + 1 for headers
    assert len(preview) == 21, "unexpected num of preview lines"
    assert len(preview[0]) == df.shape[1] + len(
        df.index.names
    ), "unexpected num of header columns"
    assert len(preview[1]) == df.shape[1] + len(
        df.index.names
    ), "unexpected num of value columns"

    features = sorted(featureset.spec.features.keys())
    stats = sorted(featureset.status.stats.keys())
    stats.remove(key)
    assert features == stats, "didnt infer stats for all features"

    stat_columns = list(featureset.status.stats["movements"].keys())
    assert stat_columns == [
        "count",
        "mean",
        "std",
        "min",
        "max",
        "hist",
    ], "wrong stats result"


def test_target_no_time_column():
    t = ParquetTarget(path="jhjhjhj")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        t.as_df(
            start_time=pd.Timestamp("2021-06-09 09:30:00"),
            end_time=pd.Timestamp("2021-06-09 10:30:00"),
        )


def test_check_permissions():
    data = pd.DataFrame(
        {
            "time_stamp": [
                pd.Timestamp("2021-06-09 09:30:06.008"),
                pd.Timestamp("2021-06-09 10:29:07.009"),
                pd.Timestamp("2021-06-09 09:29:08.010"),
            ],
            "data": [10, 20, 30],
            "string": ["ab", "cd", "ef"],
        }
    )
    data_set1 = fs.FeatureSet("fs1", entities=[Entity("string")])

    mlrun.db.FileRunDB.verify_authorization = unittest.mock.Mock(
        side_effect=mlrun.errors.MLRunAccessDeniedError("")
    )

    try:
        fs.preview(
            data_set1,
            data,
            entity_columns=[Entity("string")],
            timestamp_key="time_stamp",
        )
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass

    try:
        fs.ingest(data_set1, data, infer_options=fs.InferOptions.default())
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass

    features = ["fs1.*"]
    feature_vector = fs.FeatureVector("test", features)
    try:
        fs.get_offline_features(feature_vector, entity_timestamp_column="time_stamp")
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass

    try:
        fs.get_online_feature_service(feature_vector)
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass

    try:
        fs.deploy_ingestion_service(featureset=data_set1)
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass

    try:
        data_set1.purge_targets()
        assert False
    except mlrun.errors.MLRunAccessDeniedError:
        pass


def test_check_timestamp_key_is_entity():
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fs.FeatureSet(
            "imp1", entities=[Entity("time_stamp")], timestamp_key="time_stamp"
        )
