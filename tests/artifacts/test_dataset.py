import numpy
import pandas
import pandas.io.json
import pathlib
import tests.conftest

import mlrun.artifacts.dataset


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


def test_dataset_upload():
    """
    This test fails when we use numpy>=1.20 and is here to reproduce the scenario that didn't work
    which caused us to upbound numpy to 1.20
    see https://github.com/Azure/MachineLearningNotebooks/issues/1314
    """
    data_frame = pandas.DataFrame({"x": [1, 2]})
    target_path = pathlib.Path(tests.conftest.results) / "dataset"
    artifact = mlrun.artifacts.dataset.DatasetArtifact(
        df=data_frame, target_path=str(target_path)
    )
    artifact.upload()
