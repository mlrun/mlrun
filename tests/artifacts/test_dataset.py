import pathlib

import dask.dataframe as dd
import numpy
import pandas
import pandas.io.json
from dask.distributed import Client

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
        df=data_frame, target_path=str(target_path), format="csv",
    )
    data_frame.to_csv(src_path)
    artifact.src_path = src_path
    artifact.upload()
    assert artifact.hash is not None


def _generate_dataset_artifact(format_):
    data_frame = pandas.DataFrame({"x": [1, 2]})
    target_path = pathlib.Path(tests.conftest.results) / "dataset"
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame, target_path=str(target_path), format=format_,
    )
    return artifact


def test_dataset_preview_size_limit_from_large_dask_dataframe():
    client = Client()  # noqa: F841
    print("Creating dataframes > 1GB")
    A = numpy.random.random_sample(size=(25000000, 6))
    df = pandas.DataFrame(data=A, columns=list("ABCDEF"))
    ddf = dd.from_pandas(df, npartitions=10)
    print("Created ddf")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview) == mlrun.artifacts.dataset.default_preview_rows_length

    # override limit
    limit = 25
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf, preview=limit)
    assert len(artifact.preview) == limit
    print("passed large ddf assertion limit")

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    # For a large DDF(>1GB in-memory), sample 20% of the rows for
    # creating the preview
    assert len(artifact.preview) == (len(ddf) * 0.2)
    print("passed .2 length assertion")

    # more than allowed columns
    number_of_columns = mlrun.artifacts.dataset.max_preview_columns * 3
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(500000, number_of_columns)),
        columns=list(range(number_of_columns)),
    )
    ddf = dd.from_pandas(data_frame, npartitions=4)
    ddf = ddf.repartition(partition_size="100MB")
    print("created ddf2")
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=ddf)
    assert len(artifact.preview[0]) == mlrun.artifacts.dataset.max_preview_columns
    print("max_preview_columns")
    assert artifact.stats is None
    print("artifacts.stats passed")

    # ignore limits
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=ddf, ignore_preview_limits=True
    )
    assert len(artifact.preview[0]) == number_of_columns + 1
    print("Ignore preview limits")

    # too many rows for stats computation
    data_frame = pandas.DataFrame(
        numpy.random.randint(0, 10, size=(mlrun.artifacts.dataset.max_csv * 3, 1)),
        columns=["A"],
    )
    ddf = dd.from_pandas(data_frame, npartitions=2)
    ddf = ddf.repartition(partition_size="100MB").persist()
    artifact = mlrun.artifacts.dataset.DatasetArtifact(df=data_frame)
    assert artifact.stats is None
    print("too many rows for stats computation")


def test_dataset_preview_size_limit_from_small_dask_dataframe():
    print("Creating client...")
    client = Client()
    print(f"Scheduler at:  {client.scheduler_info()['address']}")
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
