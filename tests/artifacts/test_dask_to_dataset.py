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
    monkeypatch.setattr(mlrun, "artifacts.dataset.max_ddf_size", 0.3)
    # mlrun.artifacts.dataset.max_ddf_size = 0.3

    print("Creating dataframe and setting memory limit")
    A = numpy.random.random_sample(size=(9000000, 6))
    df = pandas.DataFrame(data=A, columns=list("ABCDEF"))
    print("Verify the memory size of the dataframe is >400MB")
    assert (df.memory_usage().sum() // 1e6) > 400
    ddf = dd.from_pandas(df, npartitions=10)
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

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(200000, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="100MB")
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


def test_dataset_preview_size_limit_from_small_dask_dataframe(make_client):
    A = numpy.random.random_sample(size=(10000, 6))
    df = pandas.DataFrame(data=A, columns=list("ABCDEF"))
    ddf = dd.from_pandas(df, npartitions=4).persist()
    print("Done making dataframe")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length
    print("passed length assertion on preview")

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf, preview=limit)
    assert len(artifact.preview) == limit
    print("passed limit override")

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    # For a small DDF (<1GB), convert to Pandas
    assert len(artifact.preview) == len(ddf)

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
