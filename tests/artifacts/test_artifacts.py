import mlrun
import mlrun.artifacts
from mlrun.artifacts.manager import extend_artifact_path
from mlrun.utils import StorePrefix


def test_artifacts_export_required_fields():
    artifact_classes = [
        mlrun.artifacts.Artifact,
        mlrun.artifacts.ChartArtifact,
        mlrun.artifacts.PlotArtifact,
        mlrun.artifacts.DatasetArtifact,
        mlrun.artifacts.ModelArtifact,
        mlrun.artifacts.TableArtifact,
    ]

    required_fields = [
        "key",
        "kind",
        "db_key",
    ]

    for artifact_class in artifact_classes:
        for required_field in required_fields:
            assert required_field in artifact_class._dict_fields


def test_artifact_uri():
    artifact = mlrun.artifacts.Artifact("data", body="abc")
    prefix, uri = mlrun.datastore.parse_store_uri(artifact.uri)
    assert prefix == StorePrefix.Artifact, "illegal artifact uri"

    artifact = mlrun.artifacts.ModelArtifact("data", body="abc")
    prefix, uri = mlrun.datastore.parse_store_uri(artifact.uri)
    assert prefix == StorePrefix.Model, "illegal artifact uri"


def test_extend_artifact_path():
    tests = ["", "./", "abc", "+/", "+/x"]
    expected = ["", "./", "abc", "", "x"]
    for i, test in enumerate(tests):
        assert extend_artifact_path(test, "") == expected[i]
    expected = ["yz", "./", "abc", "yz/", "yz/x"]
    for i, test in enumerate(tests):
        assert extend_artifact_path(test, "yz") == expected[i]
