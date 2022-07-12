import pathlib

import mlrun
import mlrun.artifacts
from mlrun.artifacts.manager import extend_artifact_path
from mlrun.utils import StorePrefix
from tests import conftest

results_dir = (pathlib.Path(conftest.results) / "artifacts").absolute()


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
        "kind",
        "metadata",
        "spec",
    ]

    required_metadata_fields = [
        "key",
    ]

    required_spec_fields = [
        "db_key",
    ]

    for artifact_class in artifact_classes:
        for required_field in required_fields:
            assert required_field in artifact_class._dict_fields
        dummy_artifact = artifact_class()
        for required_metadata_field in required_metadata_fields:
            assert required_metadata_field in dummy_artifact.metadata._dict_fields
        for required_spec_field in required_spec_fields:
            assert required_spec_field in dummy_artifact.spec._dict_fields


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


class FakeProducer:
    def __init__(self, name="", kind="run"):
        self.kind = kind
        self.name = name


def test_generate_target_path():
    Artifact = mlrun.artifacts.Artifact
    Model = mlrun.artifacts.ModelArtifact
    cases = [
        # artifact_path, artifact, src_path, iter, producer, expected
        ("x", Artifact("k1"), None, FakeProducer("j1"), "x/j1/0/k1"),
        (
            None,
            Artifact("k2", format="html"),
            1,
            FakeProducer("j1"),
            "j1/1/k2.html",
        ),
        (
            "",
            Artifact("k3", src_path="model.pkl"),
            0,
            FakeProducer("j1"),
            "j1/0/k3.pkl",
        ),
        (
            "x",
            Artifact("k4", src_path="a.b"),
            None,
            FakeProducer(kind="project"),
            "x/k4.b",
        ),
        (
            "",
            Model("k5", model_dir="y", model_file="model.pkl"),
            0,
            FakeProducer("j1"),
            "j1/0/k5/",
        ),
        (
            "x",
            Model("k6", model_file="a.b"),
            None,
            FakeProducer(kind="project"),
            "x/k6/",
        ),
    ]
    for artifact_path, artifact, iter, producer, expected in cases:
        artifact.iter = iter
        target = mlrun.artifacts.base.generate_target_path(
            artifact, artifact_path, producer
        )
        print(f"\ntarget:   {target}\nexpected: {expected}")


def test_export_import():
    project = mlrun.new_project("log-mod", save=False)
    target_project = mlrun.new_project("log-mod2", save=False)
    model = project.log_model(
        "mymod",
        body=b"123",
        model_file="model.pkl",
        extra_data={"kk": b"456"},
        artifact_path=results_dir,
    )

    for suffix in ["yaml", "json", "zip"]:
        # export the artifact to a file
        model.export(f"{results_dir}/a.{suffix}")

        # import and log the artifact to the new project
        artifact = target_project.import_artifact(
            f"{results_dir}/a.{suffix}", f"mod-{suffix}", artifact_path=results_dir
        )
        assert artifact.kind == "model"
        assert artifact.metadata.key == f"mod-{suffix}"
        assert artifact.metadata.project == "log-mod2"
        temp_path, model_spec, extra_dataitems = mlrun.artifacts.get_model(artifact.uri)
        with open(temp_path, "rb") as fp:
            data = fp.read()
        assert data == b"123"
        assert extra_dataitems["kk"].get() == b"456"


def test_inline_body():
    project = mlrun.new_project("inline", save=False)

    # log an artifact and save the content/body in the object (inline)
    artifact = project.log_artifact(
        "x", body="123", is_inline=True, artifact_path=results_dir
    )
    assert artifact.spec.get_body() == "123"
    artifact.export(f"{results_dir}/x.yaml")

    # verify the body survived the export and import
    artifact = project.import_artifact(
        f"{results_dir}/x.yaml", "y", artifact_path=results_dir
    )
    assert artifact.spec.get_body() == "123"
    assert artifact.metadata.key == "y"
