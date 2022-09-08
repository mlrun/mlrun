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
import pathlib

import numpy
import pandas
import pandas.io.json

import mlrun.artifacts.dataset
import tests.conftest


def test_dataset_preview_size_limit():
    # more than allowed rows
    data_frame = pandas.DataFrame(
        range(0, mlrun.artifacts.dataset.default_preview_rows_length * 2), columns=["A"]
    )
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame, preview=limit)
    assert len(artifact.preview) == limit

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame, ignore_preview_limits=True
    )
    assert len(artifact.preview) == len(data_frame)

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(10, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert len(artifact.preview[0]) == mlrun.artifacts.dataset.max_preview_columns
    assert artifact.stats is None

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame, ignore_preview_limits=True
    )
    assert len(artifact.preview[0]) == number_of_columns + 1

    # too many rows for stats computation
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(mlrun.artifacts.dataset.max_csv * 3, 1)),
        columns=["A"],
    )
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


def _generate_dataset_artifact(format_):
    data_frame = pandas.DataFrame({"x": [1, 2]})
    target_path = pathlib.Path(tests.conftest.results) / "dataset"
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame,
        target_path=str(target_path),
        format=format_,
    )
    return artifact
