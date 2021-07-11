import mlrun
import mlrun.artifacts
from tests.conftest import out_path, rundb_path


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
    mlrun.mlconf.dbpath = rundb_path
    context = mlrun.get_or_create_ctx("test-artifact")
    artifact = context.log_artifact("data", body="abc", artifact_path=out_path)

    prefix, uri = mlrun.datastore.parse_store_uri(artifact.uri)
    assert prefix == "artifacts", "illegal artifact uri"
    assert artifact.dataitem.get(encoding="utf-8") == "abc", "wrong .dataitem result"
