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
import pathlib

import dask.dataframe as dd
import numpy
import pandas
import pandas.io.json
import pytest

import mlrun.artifacts.dataset
import tests.conftest


def test_dataset_preview_size_limit():
    # more than allowed rows
    data_frame = pandas.DataFrame(
        range(0, mlrun.artifacts.dataset.default_preview_rows_length * 2), columns=["A"]
    )
    _assert_data_artifact_limits(data_frame, len(data_frame))

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(10, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    _assert_data_artifacts(data_frame, number_of_columns)
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None


def test_dataset_upload_parquet():
    """
    This test fails when we use numpy>=1.20 and is here to reproduce the scenario that didn't work
    which caused us to upbound numpy to 1.20
    see https://github.com/Azure/MachineLearningNotebooks/issues/1314
    """
    artifact = _generate_dataset_artifact(format_="parquet")
    artifact.upload()


def test_dataset_upload_csv():
    """
    This test fails when we use pandas<1.2 and is here to reproduce the scenario that didn't work
    which caused us to downbound pandas to 1.2
    see https://pandas.pydata.org/docs/whatsnew/v1.2.0.html#support-for-binary-file-handles-in-to-csv
    """
    artifact = _generate_dataset_artifact(format_="csv")
    artifact.upload()


def test_dataset_upload_with_src_path_filling_hash():
    data_frame = pandas.DataFrame({"x": [1, 2]})
    src_path = pathlib.Path(tests.conftest.results) / "dataset"
    target_path = pathlib.Path(tests.conftest.results) / "target-dataset"
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame,
        target_path=str(target_path),
        format="csv",
    )
    data_frame.to_csv(src_path)
    artifact.src_path = src_path
    artifact.upload()
    assert artifact.hash is not None


def test_dataset_upload_without_df_or_body():
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        target_path=str(pathlib.Path(tests.conftest.results) / "target-dataset"),
        format="csv",
    )
    # make sure uploading doesn't fail
    artifact.upload()
    assert artifact.hash is None
    assert artifact.size is None


def test_resolve_dataset_hash_path():
    for test_case in [
        {
            "artifact": mlrun.artifacts.dataset.DatasetArtifact(
                df=pandas.DataFrame({"x": [1, 2]}),
                format="csv",
            ),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "0d1c62a76b705b34bb70f355162f83402f3640e3",
            "expected_file_target": "v3io://just/regular/path/0d1c62a76b705b34bb70f355162f83402f3640e3.csv",
            "expected_error": None,
        },
        {
            # generates incremental values dataframe
            "artifact": mlrun.artifacts.dataset.DatasetArtifact(
                df=pandas.DataFrame(
                    numpy.broadcast_to(numpy.arange(1, 300 + 1)[:, None], (300, 100))
                ),
                format="parquet",
            ),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "f039fcf3a8b4bd6805b2bec0c6db96c2189eb9e2",
            "expected_file_target": "v3io://just/regular/path/f039fcf3a8b4bd6805b2bec0c6db96c2189eb9e2.parquet",
            "expected_error": None,
        },
        {
            # without format
            "artifact": mlrun.artifacts.dataset.DatasetArtifact(
                df=pandas.DataFrame({"x": [1, 2]}),
            ),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "0d1c62a76b705b34bb70f355162f83402f3640e3",
            "expected_file_target": "v3io://just/regular/path/0d1c62a76b705b34bb70f355162f83402f3640e3",
            "expected_error": None,
        },
        {
            # without artifact_path, expected to fail
            "artifact": mlrun.artifacts.dataset.DatasetArtifact(
                df=pandas.DataFrame({"x": [1, 2]}),
            ),
            "artifact_path": None,
            "expected_hash": None,
            "expected_file_target": None,
            "expected_error": mlrun.errors.MLRunInvalidArgumentError,
        },
        # tests dask dataframe
        {
            "artifact": mlrun.artifacts.dataset.DatasetArtifact(
                df=dd.from_pandas(pandas.DataFrame({"x": [1, 2]}), npartitions=1),
                format="csv",
            ),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "0d1c62a76b705b34bb70f355162f83402f3640e3",
            "expected_file_target": "v3io://just/regular/path/0d1c62a76b705b34bb70f355162f83402f3640e3.csv",
            "expected_error": None,
        },
    ]:
        artifact = test_case.get("artifact")
        artifact_path = test_case.get("artifact_path")
        expected_error: mlrun.errors.MLRunBaseError = test_case.get(
            "expected_error", None
        )
        if expected_error:
            with pytest.raises(expected_error):
                artifact.resolve_dataframe_target_hash_path(
                    dataframe=artifact.df, artifact_path=artifact_path
                )
            break
        dataset_hash, target_path = artifact.resolve_dataframe_target_hash_path(
            dataframe=artifact.df, artifact_path=artifact_path
        )
        assert test_case.get("expected_hash") == dataset_hash
        assert test_case.get("expected_file_target") == target_path


def test_dataset_stats():
    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "age": [42, 52, 36, 24, 73],
        "testScore": [25, 94, 57, 62, 70],
    }
    df = pandas.DataFrame(
        raw_data, columns=["first_name", "last_name", "age", "testScore"]
    )
    for test_case in [
        {
            # status is not set
            "df": df,
            "stats": None,
            "expected_none_status_stats": False,
        },
        {
            # status is set to True
            "df": df,
            "stats": True,
            "expected_none_status_stats": False,
        },
        {
            # status is set to False
            "df": df,
            "stats": False,
            "expected_none_status_stats": True,
        },
        {
            # status is not set and df is very large
            "df": pandas.DataFrame(
                raw_data, columns=[f"column-title-{i}" for i in range(200)]
            ),
            "stats": None,
            "expected_none_status_stats": True,
        },
    ]:
        dataset_artifact = mlrun.artifacts.dataset.DatasetArtifact(
            df=test_case.get("df"), stats=test_case.get("stats")
        )

        if test_case["expected_none_status_stats"]:
            assert dataset_artifact.status.stats is None
        else:
            assert dataset_artifact.status.stats is not None


def test_get_log_dataset_dont_duplicate_index_column():
    source_url = mlrun.get_sample_path("data/iris/iris.data.raw.csv")
    df = mlrun.get_dataitem(source_url).as_df()
    artifact = mlrun.get_or_create_ctx("test").log_dataset("iris", df=df, upload=False)
    index_counter = 0
    for field in artifact.spec.schema["fields"]:
        if field["name"] == "index":
            index_counter += 1
    assert index_counter == 1

    df = df.set_index("label")
    artifact = mlrun.get_or_create_ctx("test").log_dataset("iris", df=df, upload=False)
    index_counter = 0
    for field in artifact.spec.schema["fields"]:
        if field["name"] == "index":
            index_counter += 1
    assert index_counter == 1


def test_log_dataset_with_column_overflow(monkeypatch):
    context = mlrun.get_or_create_ctx("test")
    source_url = mlrun.get_sample_path("data/iris/iris.data.raw.csv")
    df = mlrun.get_dataitem(source_url).as_df()

    monkeypatch.setattr(mlrun.artifacts.dataset, "max_preview_columns", 10)
    artifact = context.log_dataset("iris", df=df, upload=False)
    assert len(artifact.spec.header) == 6
    assert artifact.status.header_original_length == 6

    monkeypatch.setattr(mlrun.artifacts.dataset, "max_preview_columns", 2)
    artifact = context.log_dataset("iris", df=df, upload=False)
    assert len(artifact.spec.header) == 2
    assert artifact.status.header_original_length == 6


def test_create_dataset_non_existing_label():
    project = mlrun.new_project("artifact-experiment", save=False)
    df = pandas.DataFrame(
        {
            "column_1": [0, 1, 2, 3, 4],
            "column_2": [5, 6, 7, 8, 9],
        }
    )

    with pytest.raises(mlrun.errors.MLRunValueError):
        project.log_dataset("my_dataset", df=df, label_column="column_3")


def test_dataset_preview_size_limit_from_large_dask_dataframe(monkeypatch):
    """
    To simplify testing the behavior of a large Dask DataFrame as a mlrun
    Dataset, we set the default max_ddf_size parameter at 300MB,
    and test with dataframes of 430MB in size.  Default behavior is to
    convert any Dask DataFrames of size <1GB to Pandas, else use Dask to create the artifact.
    """
    # Set a MAX_DDF_SIZE param to simplify testing
    monkeypatch.setattr(mlrun.artifacts.dataset, "max_ddf_size", 0.001)

    print("Creating dataframe and setting memory limit")
    data = numpy.random.random_sample(size=(50000, 6))
    df = pandas.DataFrame(data=data, columns=list("ABCDEF"))
    print("Verify the memory size of the dataframe is >400MB")
    assert (df.memory_usage().sum() // 1e3) > 200
    ddf = dd.from_pandas(df, npartitions=4)

    # for a large DDF, sample 20% of the rows for
    # creating the preview
    _assert_data_artifact_limits(
        ddf, len(ddf.sample(frac=0.2).compute().values.tolist())
    )

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(2000, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="1MB")
    _assert_data_artifacts(ddf, number_of_columns)

    ddf = dd.from_pandas(data_frame, npartitions=2)
    ddf.repartition(partition_size="100MB")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None


def test_dataset_preview_size_limit_from_small_dask_dataframe():
    print("Starting preview for small dask dataframe")
    df_data = numpy.random.random_sample(size=(100, 6))
    df = pandas.DataFrame(data=df_data, columns=list("ABCDEF"))
    ddf = dd.from_pandas(df, npartitions=4).persist()
    _assert_data_artifact_limits(ddf, len(df))

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(10, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="100MB").persist()
    _assert_data_artifacts(ddf, number_of_columns)
    # too many rows for stats computation
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(mlrun.artifacts.dataset.max_csv * 3, 1)),
        columns=["A"],
    )
    ddf = dd.from_pandas(data_frame, npartitions=2)
    ddf.repartition(partition_size="100MB").persist()
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None


def _generate_dataset_artifact(format_):
    data_frame = pandas.DataFrame({"x": [1, 2]})
    target_path = pathlib.Path(tests.conftest.results) / "dataset"
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame,
        target_path=str(target_path),
        format=format_,
    )
    return artifact


def _assert_data_artifact_limits(df, expected_preview_length):
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=df)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=df, preview=limit)
    assert len(artifact.preview) == limit

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=df, ignore_preview_limits=True
    )
    assert len(artifact.preview) == expected_preview_length


def _assert_data_artifacts(df, number_of_columns):
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=df)
    assert len(artifact.preview[0]) == mlrun.artifacts.dataset.max_preview_columns
    assert artifact.stats is None

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=df, ignore_preview_limits=True
    )
    # Dataset previews have an extra column called "index"
    assert len(artifact.preview[0]) - 1 == number_of_columns
