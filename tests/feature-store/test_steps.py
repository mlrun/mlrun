# Copyright 2023 Iguazio
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
import tempfile
import time
import unittest.mock

import numpy as np
import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fstore
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
    }
    return event


def test_set_event_meta(rundb_mock):
    function = mlrun.new_function("test1", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(id_path="myid", key_path="mykey")).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    event = {"myid": "34", "mykey": "123"}
    resp = server.test(body=event)
    server.wait_for_completion()
    assert resp == {
        "id": "34",
        "key": "123",
    }


def test_set_event_random_id(rundb_mock):
    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(random_id=True)).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    resp = server.test(body={"data": "123"}, event_id="XYZ")
    server.wait_for_completion()
    assert resp["id"] != "XYZ", "id was not overwritten"


@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
@pytest.mark.parametrize("set_index_before", [True, False, 0])
def test_pandas_step_onehot(rundb_mock, entities, set_index_before):
    data, _ = get_data()
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # One Hot Encode the newly defined mappings (mapping values not unique for testing)
    one_hot_encoder_mapping = {
        "department": data["department"].tolist(),
    }
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
        engine="pandas",
    )

    # Pre-processing graph steps
    data_set_pandas.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    if len(entities) == 1:
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
            index=data["id"].values,
        )
        # pandas 2 assert_frame_equal actually checks the index name(s), so we need it
        data_ref.index.name = "id"
        data_ref.index.names = ["id"]
    else:
        data_ref = pd.DataFrame(
            {
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
            index=[data["id"].values, data["name"].values],
        )
        # pandas 2 assert_frame_equal actually checks the index name(s), so we need it
        data_ref.index.name = None
        data_ref.index.names = ["id", "name"]

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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing graph steps
    data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize("set_index_before", [True, False, 0])
@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
def test_pandas_step_imputer(rundb_mock, entities, set_index_before):
    data, data_ref = get_data(True)
    data_ref.set_index(entities, inplace=True)
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing graph steps
    data_set_pandas.graph.to(Imputer(mapping={"department": "IT"}))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing graph steps
    data_set.graph.to(Imputer(mapping={"department": "IT"}))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
@pytest.mark.parametrize("with_original", [True, False])
@pytest.mark.parametrize("set_index_before", [True, False, 0])
def test_pandas_step_mapval(rundb_mock, with_original, entities, set_index_before):
    data, _ = get_data()
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
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
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )

    if with_original:
        data_ref = data.copy()
        data_ref = data_ref.set_index(entities, drop=True)
        data_ref["age_mapped"] = ["adult", "child", "adult", "adult", "child"]
        data_ref["department_mapped"] = [1, 3, 3, 2, 1]
    else:
        age = ["adult", "child", "adult", "adult", "child"]
        department = [1, 3, 3, 2, 1]
        index = data["id"].values
        if len(entities) > 1:
            index = [data[ent].values for ent in entities]
        data_ref = pd.DataFrame({"age": age, "department": department}, index=index)
        # pandas 2 assert_frame_equal actually checks the index name(s), so we need it
        if len(entities) > 1:
            data_ref.index.name = None
            data_ref.index.names = entities
        else:
            data_ref.index.name = "id"
            data_ref.index.names = ["id"]

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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing graph steps
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
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
@pytest.mark.parametrize("set_index_before", [True, False, 0])
@pytest.mark.parametrize("timestamp_col", [None, "timestamp"])
def test_pandas_step_data_extractor(
    rundb_mock, entities, set_index_before, timestamp_col
):
    data, _ = get_data()
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing graph steps
    data_set_pandas.graph.to(
        DateExtractor(
            parts=["hour", "day_of_week"],
            timestamp_col=timestamp_col,
        )
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index(entities, drop=True)
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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing grpah steps
    data_set.graph.to(
        DateExtractor(
            parts=["hour", "day_of_week"],
            timestamp_col=timestamp_col,
        )
    )
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize(
    "mapping",
    [
        {"age": {"ranges": {"one": [0, 30], "two": ["a", "inf"]}}},
        {"names": {"A": 1, "B": False}},
    ],
)
def test_mapvalues_mixed_types_validator(rundb_mock, mapping):
    data, _ = get_data()
    data_to_ingest = data.copy()
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(
        MapValues(
            mapping=mapping,
            with_original_features=True,
        )
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=f"^MapValues - mapping values of the same column must be in the same type, which was not the case for"
        f" Column '{list(mapping.keys())[0]}'$",
    ):
        data_set_pandas.ingest(
            data_to_ingest,
            targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
        )


def test_mapvalues_combined_mapping_validator(rundb_mock):
    data, _ = get_data()
    data_to_ingest = data.copy()
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity("id")],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(
        MapValues(
            mapping={
                "age": {"ranges": {"one": [0, 30], "two": ["a", "inf"]}, 4: "kid"}
            },
            with_original_features=True,
        )
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="^MapValues - mapping values of the same column can not combine ranges and single "
        "replacement, which is the case for column 'age'$",
    ):
        data_set_pandas.ingest(
            data_to_ingest,
            targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
        )


@pytest.mark.parametrize("set_index_before", [True, False, 0])
@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
def test_pandas_step_data_validator(rundb_mock, entities, set_index_before):
    data, _ = get_data()
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing grpah steps
    data_set_pandas.graph.to(FeaturesetValidator())
    data_set_pandas["age"] = fstore.Feature(
        validator=MinMaxValidator(min=30, severity="info"),
        value_type=fstore.ValueType.INT16,
    )
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index(entities, drop=True)
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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing graph steps
    data_set.graph.to(FeaturesetValidator())
    data_set["age"] = fstore.Feature(
        validator=MinMaxValidator(min=30, severity="info"),
        value_type=fstore.ValueType.INT16,
    )
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize("set_index_before", [True, False, 0])
@pytest.mark.parametrize("entities", [["id"], ["id", "name"]])
def test_pandas_step_drop_feature(rundb_mock, entities, set_index_before):
    data, _ = get_data()
    data_to_ingest = data.copy()
    if set_index_before or len(entities) == 1:
        data_to_ingest.set_index(entities, inplace=True)
    elif isinstance(set_index_before, int) and len(entities) > 1:
        data_to_ingest.set_index(entities[set_index_before], inplace=True)
    # Define the corresponding FeatureSet
    data_set_pandas = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
        engine="pandas",
    )
    # Pre-processing graph steps
    data_set_pandas.graph.to(DropFeatures(features=["age"]))
    data_set_pandas._run_db = rundb_mock

    data_set_pandas.reload = unittest.mock.Mock()
    data_set_pandas.save = unittest.mock.Mock()
    data_set_pandas.purge_targets = unittest.mock.Mock()
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Ingest our dataset through our defined pipeline
    df_pandas = data_set_pandas.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
    data_ref = data.copy()
    data_ref = data_ref.set_index(entities, drop=True)
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
    data_set = fstore.FeatureSet(
        "fs-new",
        entities=[fstore.Entity(ent) for ent in entities],
        description="feature set",
    )
    # Pre-processing graph steps
    data_set.graph.to(DropFeatures(features=["age"]))
    data_set._run_db = rundb_mock

    data_set.reload = unittest.mock.Mock()
    data_set.save = unittest.mock.Mock()
    data_set.purge_targets = unittest.mock.Mock()

    # Ingest our dataset through our defined pipeline
    df = data_set.ingest(
        data_to_ingest,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
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


@pytest.mark.parametrize("engine", ["storey", "pandas"])
def test_imputer_default_value(rundb_mock, engine):
    data_with_nones = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "height": [None, 160, pd.NA, np.nan],
            "age": [20, pd.NaT, 19, 18],
        }
    )
    # Building graph with Imputer:
    feature_set = fstore.FeatureSet(
        "fs-default-value",
        entities=["id"],
        description="feature set with nones",
        engine=engine,
    )
    feature_set.graph.to(Imputer(default_value=1))

    # Mocking
    output_path = tempfile.TemporaryDirectory()
    feature_set._run_db = rundb_mock
    feature_set.reload = unittest.mock.Mock()
    feature_set.save = unittest.mock.Mock()
    feature_set.purge_targets = unittest.mock.Mock()

    imputed_df = feature_set.ingest(
        source=data_with_nones,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )

    # Checking that the ingested dataframe is none-free:
    assert not imputed_df.isnull().values.any()


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
                "id": ["a", "v", "h", "g", "j"],
            },
        )
        department = [None, "RD", "RD", "Marketing", "IT"]
    data = pd.DataFrame(
        {
            "name": names,
            "age": ages,
            "department": department,
            "timestamp": timestamp,
            "id": ["a", "v", "h", "g", "j"],
        },
    )
    return data, data_ref


# ML-7868
@pytest.mark.parametrize("engine", ["storey", "pandas"])
def test_parquet_source_with_category(rundb_mock, engine):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
        }
    )
    df["my_category"] = df["id"].astype("category")
    feature_set = fstore.FeatureSet(
        "fs-default-value",
        entities=["id"],
        engine=engine,
    )

    # Mocking
    output_path = tempfile.TemporaryDirectory()
    feature_set._run_db = rundb_mock
    feature_set.reload = unittest.mock.Mock()
    feature_set.save = unittest.mock.Mock()
    feature_set.purge_targets = unittest.mock.Mock()

    feature_set.ingest(
        source=df,
        targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
    )
