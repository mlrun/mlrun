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

import pytest

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
        assert target == expected


def assets_path():
    return pathlib.Path(__file__).absolute().parent / "assets"


def test_resolve_file_hash_path():
    for test_case in [
        {
            "artifact": mlrun.artifacts.Artifact("results", src_path="results.csv"),
            "src_path": str(assets_path() / "results.csv"),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "4697a8195a0e8ef4e1ee3119268337c8e0afabfc",
            "expected_file_target": "v3io://just/regular/path/4697a8195a0e8ef4e1ee3119268337c8e0afabfc.csv",
            "expected_error": None,
        },
        # expected to fail because no artifact has been provided
        {
            "artifact": mlrun.artifacts.Artifact("results", src_path="results.csv"),
            "src_path": str(assets_path() / "results.csv"),
            "artifact_path": None,
            "expected_hash": None,
            "expected_file_target": None,
            "expected_error": mlrun.errors.MLRunInvalidArgumentError,
        },
    ]:
        mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = test_case.get(
            "generate_target_path_from_artifact_hash"
        )
        mlrun.mlconf.artifacts.calculate_hash = test_case.get("calculate_hash")
        artifact = test_case.get("artifact")
        src_path = test_case.get("src_path")
        artifact_path = test_case.get("artifact_path")
        expected_error: mlrun.errors.MLRunBaseError = test_case.get(
            "expected_error", None
        )
        if expected_error:
            with pytest.raises(expected_error):
                artifact.resolve_file_target_hash_path(
                    src=src_path, artifact_path=artifact_path
                )
            break
        file_hash, target_path = artifact.resolve_file_target_hash_path(
            src=src_path, artifact_path=artifact_path
        )
        assert test_case.get("expected_file_target") == target_path
        assert test_case.get("expected_hash") == file_hash


def test_resolve_body_hash_path():
    for test_case in [
        {
            "artifact": mlrun.artifacts.Artifact("results", body="asdasdasdasdas"),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "expected_file_target": "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "expected_error": None,
        },
        {
            "artifact": mlrun.artifacts.Artifact("results", body=b"asdasdasdasdas"),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "expected_file_target": "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "expected_error": None,
        },
        {
            "artifact": mlrun.artifacts.Artifact("results", body={"ba": "nana"}),
            "artifact_path": "v3io://just/regular/path",
            "expected_hash": None,
            "expected_file_target": None,
            "expected_error": TypeError,
        },
        # expected to fail because no artifact has been provided
        {
            "artifact": mlrun.artifacts.Artifact("results", body="asdasdasdasdas"),
            "artifact_path": None,
            "expected_hash": None,
            "expected_file_target": None,
            "expected_error": mlrun.errors.MLRunInvalidArgumentError,
        },
    ]:
        artifact = test_case.get("artifact")
        artifact_path = test_case.get("artifact_path")
        expected_error: mlrun.errors.MLRunBaseError = test_case.get(
            "expected_error", None
        )
        if expected_error:
            with pytest.raises(expected_error):
                artifact.resolve_body_target_hash_path(
                    body=artifact.get_body(), artifact_path=artifact_path
                )
            break
        body_hash, target_path = artifact.resolve_body_target_hash_path(
            body=artifact.get_body(), artifact_path=artifact_path
        )
        assert test_case.get("expected_file_target") == target_path
        assert test_case.get("expected_hash") == body_hash


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
