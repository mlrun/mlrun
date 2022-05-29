import pandas
import pytest

import mlrun.artifacts.dataset


# Verify TableArtifact.get_body() doesn't hang (ML-2184)
@pytest.mark.parametrize("use_dataframe", [True, False])
def test_table_artifact_get_body(use_dataframe):
    artifact = _generate_table_artifact(use_dataframe=use_dataframe)

    artifact_body = artifact.get_body()
    print("Artifact body:\n" + artifact_body)
    assert artifact_body is not None


def _generate_table_artifact(use_dataframe=True):
    if use_dataframe:
        data_frame = pandas.DataFrame({"x": [1, 2]})
    else:
        body = "just an artifact body"

    artifact = mlrun.artifacts.dataset.TableArtifact(
        df=data_frame if use_dataframe else None,
        body=body if not use_dataframe else None,
    )
    return artifact
