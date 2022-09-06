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
import dask.dataframe as dd
import numpy
import pandas

import mlrun.artifacts.dataset


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
    A = numpy.random.random_sample(size=(50000, 6))
    df = pandas.DataFrame(data=A, columns=list("ABCDEF"))
    print("Verify the memory size of the dataframe is >400MB")
    assert (df.memory_usage().sum() // 1e3) > 200
    ddf = dd.from_pandas(df, npartitions=4)
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf, preview=limit)
    assert len(artifact.preview) == limit

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    # For a large DDF, sample 20% of the rows for
    # creating the preview
    assert len(artifact.preview) == len(ddf.sample(frac=0.2).compute().values.tolist())
    print("Passed assertion on preview")

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(2000, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="1MB")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview[0]) == mlrun.artifacts.dataset.max_preview_columns
    assert artifact.stats is None

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    assert len(artifact.preview[0]) == number_of_columns + 1

    # too many rows for stats computation
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(mlrun.artifacts.dataset.max_csv * 3, 1)),
        columns=["A"],
    )
    ddf = dd.from_pandas(data_frame, npartitions=2)
    ddf = ddf.repartition(partition_size="100MB")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None


def test_dataset_preview_size_limit_from_small_dask_dataframe():
    print("Starting preview for small dask dataframe")
    A = numpy.random.random_sample(size=(100, 6))
    df = pandas.DataFrame(data=A, columns=list("ABCDEF"))
    ddf = dd.from_pandas(df, npartitions=4).persist()
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf, preview=limit)
    assert len(artifact.preview) == limit

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    # For a small DDF (<1GB), convert to Pandas
    assert len(artifact.preview) == len(ddf)
    print("passed length assertion on preview")

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(10, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="100MB").persist()
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview[0]) == mlrun.artifacts.dataset.max_preview_columns
    assert artifact.stats is None

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    assert len(artifact.preview[0]) == number_of_columns + 1

    # too many rows for stats computation
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(mlrun.artifacts.dataset.max_csv * 3, 1)),
        columns=["A"],
    )
    ddf = dd.from_pandas(data_frame, npartitions=2)
    ddf = ddf.repartition(partition_size="100MB").persist()
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None
