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
import datetime
import tempfile
import time
import unittest.mock

import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fs
from mlrun.datastore.targets import ParquetTarget
from mlrun.feature_store.steps import (
    DateExtractor,
    DropFeatures,
    FeaturesetValidator,
    Imputer,
    MapValues,
    OneHotEncoder,
    SetEventMetadata,
)
from mlrun.features import MinMaxValidator


def extract_meta(event):
    event.body = {
        "id": event.id,
        "key": event.key,
        "time": event.time,
    }
    return event


def test_set_event_meta():
    function = mlrun.new_function("test1", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(id_path="myid", key_path="mykey", time_path="mytime")).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    event = {"myid": "34", "mykey": "123", "mytime": "2022-01-18 15:01"}
    resp = server.test(body=event)
    server.wait_for_completion()
    assert resp == {
        "id": "34",
        "key": "123",
        "time": datetime.datetime(2022, 1, 18, 15, 1),
    }


def test_set_event_random_id():
    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(random_id=True)).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    resp = server.test(body={"data": "123"}, event_id="XYZ")
    server.wait_for_completion()
    assert resp["id"] != "XYZ", "id was not overwritten"


def test_pandas_step_onehot(rundb_mock):
    data, _ = get_data()
    # One Hot Encode the newly defined mappings
    one_hot_encoder_mapping = {
        "department": list(data["department"].unique()),
    }
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )

    data_ref = pd.DataFrame(
        {
            "name": data["name"].values,
            "age": data["age"].values,
            "department_IT": [1, 0, 0, 0, 1],
            "department_RD": [0, 1, 1, 0, 0],
            "department_Marketing": [0, 0, 0, 1, 0],
            "timestamp": [
                time.time(),
                time.time(),
                time.time(),
                time.time(),
                time.time(),
            ],
        },
        index=[0, 1, 2, 3, 5],
    )

    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )

    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


def test_pandas_step_imputer(rundb_mock):
    data, data_ref = get_data(True)
    data_ref.set_index('id', inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(Imputer(mapping={"department": "IT"}))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )

    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )

    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(Imputer(mapping={"department": "IT"}))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


@pytest.mark.parametrize("with_original", [True, False])
def test_pandas_step_mapval(rundb_mock, with_original):
    data, _ = get_data()
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(
        MapValues(
            mapping={
                "age": {"ranges": {"child": [0, 30], "adult": [30, "inf"]}},
                "department": {"IT": 1, "Marketing": 2, "RD": 3},
            },
            with_original_features=with_original,
        )
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our  dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )

    if with_original:
        data_ref = data.copy()
        data_ref = data_ref.set_index("id", drop=True)
        data_ref["age_mapped"] = ["adult", "child", "adult", "adult", "child"]
        data_ref["department_mapped"] = [1, 3, 3, 2, 1]
    else:
        age = ["adult", "child", "adult", "adult", "child"]
        department = [1, 3, 3, 2, 1]
        data_ref = pd.DataFrame(
            {"age": age, "department": department}, index=[0, 1, 2, 3, 5]
        )

    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )
    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(
        MapValues(
            mapping={
                "age": {"ranges": {"child": [0, 30], "adult": [30, "inf"]}},
                "department": {"IT": 1, "Marketing": 2, "RD": 3},
            },
            with_original_features=with_original,
        )
    )
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


def test_pandas_step_data_extractor(rundb_mock):
    data, _ = get_data()
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(
        DateExtractor(
            parts=["hour", "day_of_week"],
            timestamp_col="timestamp",
        )
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index("id", drop=True)
    data_ref["timestamp_day_of_week"] = [3, 3, 3, 3, 3]
    data_ref["timestamp_hour"] = [0, 0, 0, 0, 0]
    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )
    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(
        DateExtractor(
            parts=["hour", "day_of_week"],
            timestamp_col="timestamp",
        )
    )
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


def test_pandas_step_data_validator(rundb_mock):
    data, _ = get_data()
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(FeaturesetValidator())
    data_set_pandas["age"] = fs.Feature(
        validator=MinMaxValidator(min=30, severity="info"),
        value_type=fs.ValueType.INT16,
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index("id", drop=True)
    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )
    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(FeaturesetValidator())
    data_set["age"] = fs.Feature(
        validator=MinMaxValidator(min=30, severity="info"),
        value_type=fs.ValueType.INT16,
    )
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


def test_pandas_step_drop_feature(rundb_mock):
    data, _ = get_data()
    # Define the corresponding FeatureSet
    data_set_pandas = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(DropFeatures(features=["age"]))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = fs.ingest(
        data_set_pandas,
        data,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index("id", drop=True)
    data_ref.drop(columns=["age"], inplace=True)
    assert isinstance(df_pandas, pd.DataFrame)
    pd.testing.assert_frame_equal(
        df_pandas,
        data_ref,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )
    # Define the corresponding FeatureSet
    data_set = fs.FeatureSet(
        "fs-new",
        entities=[fs.Entity("id")],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(DropFeatures(features=["age"]))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = fs.ingest(
        data_set, data, targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")]
    )
    pd.testing.assert_frame_equal(
        df,
        df_pandas,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_like=True,
        check_names=True,
    )


def get_data(with_none=False):
    names = ["A", "B", "C", "D", "E"]
    ages = [33, 4, 76, 90, 24]
    timestamp = [time.time(), time.time(), time.time(), time.time(), time.time()]
    department = ["IT", "RD", "RD", "Marketing", "IT"]
    data_ref = None
    if with_none:
        data_ref = pd.DataFrame(
            {
                "name": names,
                "age": ages,
                "department": department,
                "timestamp": timestamp,
                'id': [0, 1, 2, 3, 5]
            },
        )
        department = [None, "RD", "RD", "Marketing", "IT"]
    data = pd.DataFrame(
        {"name": names, "age": ages, "department": department, "timestamp": timestamp, 'id': [0, 1, 2, 3, 5]},
    )
    return data, data_ref
