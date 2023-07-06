# Copyright 2023 Iguazio
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
import os
import pathlib
import typing
import unittest.mock
import uuid
from contextlib import nullcontext as does_not_raise

import pytest
import yaml

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


@pytest.mark.parametrize(
    "artifact_path,artifact,iter,producer,expected",
    [
        ("x", mlrun.artifacts.Artifact("k1"), None, FakeProducer("j1"), "x/j1/0/k1"),
        (
            None,
            mlrun.artifacts.Artifact("k2", format="html"),
            1,
            FakeProducer("j1"),
            "j1/1/k2.html",
        ),
        (
            "",
            mlrun.artifacts.Artifact("k3", src_path="model.pkl"),
            0,
            FakeProducer("j1"),
            "j1/0/k3.pkl",
        ),
        (
            "x",
            mlrun.artifacts.Artifact("k4", src_path="a.b"),
            None,
            FakeProducer(kind="project"),
            "x/k4.b",
        ),
        (
            "",
            mlrun.artifacts.ModelArtifact("k5", model_dir="y", model_file="model.pkl"),
            0,
            FakeProducer("j1"),
            "j1/0/k5/",
        ),
        (
            "x",
            mlrun.artifacts.ModelArtifact("k6", model_file="a.b"),
            None,
            FakeProducer(kind="project"),
            "x/k6/",
        ),
        (
            "",
            mlrun.artifacts.Artifact("k7", src_path="a.tar.gz"),
            None,
            FakeProducer(kind="project"),
            "k7.tar.gz",
        ),
    ],
)
def test_generate_target_path(artifact_path, artifact, iter, producer, expected):
    artifact.iter = iter
    target = mlrun.artifacts.base.generate_target_path(
        artifact, artifact_path, producer
    )
    print(f"\ntarget:   {target}\nexpected: {expected}")
    assert target == expected


def assets_path():
    return pathlib.Path(__file__).absolute().parent / "assets"


@pytest.mark.parametrize(
    "artifact,expected_hash,expected_target_path,artifact_path,generate_target_path,tag,expectation,artifact_is_logged",
    [
        (
            mlrun.artifacts.Artifact(key="some-artifact", body="asdasdasdasdas"),
            "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", body="asdasdasdasdas", format="parquet"
            ),
            "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3.parquet",
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", body="asdasdasdasdas", format="parquet"
            ),
            None,
            None,
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(key="some-artifact", body=b"asdasdasdasdas"),
            None,
            None,
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", src_path=str(assets_path() / "results.csv")
            ),
            "4697a8195a0e8ef4e1ee3119268337c8e0afabfc",
            "v3io://just/regular/path/4697a8195a0e8ef4e1ee3119268337c8e0afabfc.csv",
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", src_path=str(assets_path() / "results.csv")
            ),
            None,
            None,
            "v3io://just/regular/path",
            True,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", src_path=str(assets_path() / "results.csv")
            ),
            None,
            "v3io://just/regular/path/test/0/some-artifact.csv",
            "v3io://just/regular/path",
            False,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(
                key="some-artifact", body="asdasdasdasdas", format="parquet"
            ),
            None,
            "v3io://just/regular/path/test/0/some-artifact.parquet",
            "v3io://just/regular/path",
            False,
            "",
            does_not_raise(),
            True,
        ),
        (
            mlrun.artifacts.Artifact(key="some-artifact", body="asdasdasdasdas"),
            "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path",
            True,
            "valid-tag-name",
            does_not_raise(),
            True,
        ),
        (
            # test log_artifact fails when given an invalid tag, and the artifact is not logged
            mlrun.artifacts.Artifact(key="some-artifact", body="asdasdasdasdas"),
            "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "/tmp/",
            True,
            "tag_name_invalid!@#",
            pytest.raises(mlrun.errors.MLRunInvalidArgumentError),
            False,
        ),
    ],
)
def test_log_artifact(
    artifact: mlrun.artifacts.Artifact,
    expected_hash: str,
    expected_target_path: str,
    artifact_path: str,
    generate_target_path: bool,
    tag: str,
    expectation: typing.Any,
    artifact_is_logged: bool,
    monkeypatch,
):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = (
        generate_target_path
    )

    monkeypatch.setattr(
        mlrun.datastore.DataItem,
        "upload",
        lambda *args, **kwargs: unittest.mock.Mock(),
    )
    monkeypatch.setattr(
        mlrun.datastore.DataItem,
        "put",
        lambda *args, **kwargs: unittest.mock.Mock(),
    )

    with expectation:
        logged_artifact = mlrun.get_or_create_ctx("test").log_artifact(
            artifact,
            artifact_path=artifact_path,
            tag=tag,
        )

    if not expected_hash and generate_target_path:
        if artifact.get_body():
            expected_hash = mlrun.artifacts.base.calculate_blob_hash(
                artifact.get_body()
            )
        else:
            expected_hash = mlrun.utils.calculate_local_file_hash(
                artifact.spec.src_path
            )
    if artifact_is_logged:
        if artifact.spec.format:
            assert logged_artifact.target_path.endswith(f".{artifact.spec.format}")

        if expected_target_path:
            assert expected_target_path == logged_artifact.target_path

        if expected_hash:
            assert expected_hash == logged_artifact.metadata.hash
            assert expected_hash in logged_artifact.target_path


def test_log_artifact_with_target_path_and_upload_options():
    for target_path in ["s3://some/path", None]:
        # True and None expected to upload
        for upload_options in [False, True, None]:
            artifact = mlrun.artifacts.Artifact(
                key="some-artifact", body="asdasdasdasdas", format="parquet"
            )
            logged_artifact = mlrun.get_or_create_ctx("test").log_artifact(
                artifact, target_path=target_path, upload=upload_options
            )
            print(logged_artifact)
            if not target_path and (upload_options or upload_options is None):
                # if no target path was given, and upload is True or None, we expect the artifact to get uploaded
                # and a target path to be generated
                assert logged_artifact.target_path is not None
                assert logged_artifact.metadata.hash is not None
            elif target_path:
                # if target path is given, we don't upload and therefore don't calculate hash
                assert logged_artifact.target_path == target_path
                assert logged_artifact.metadata.hash is None


@pytest.mark.parametrize(
    "artifact,artifact_path,expected_hash,expected_target_path,expected_error",
    [
        (
            mlrun.artifacts.Artifact(
                "results", src_path=str(assets_path() / "results.csv")
            ),
            "v3io://just/regular/path",
            "4697a8195a0e8ef4e1ee3119268337c8e0afabfc",
            "v3io://just/regular/path/4697a8195a0e8ef4e1ee3119268337c8e0afabfc.csv",
            None,
        ),
        (
            mlrun.artifacts.Artifact(
                "results", src_path=str(assets_path() / "results.csv")
            ),
            "v3io://just/regular/path",
            None,
            None,
            None,
        ),
        (
            mlrun.artifacts.Artifact(
                "results", src_path=str(assets_path() / "results.csv")
            ),
            None,
            None,
            None,
            mlrun.errors.MLRunInvalidArgumentError,
        ),
    ],
)
def test_resolve_file_hash_path(
    artifact: mlrun.artifacts.Artifact,
    artifact_path: str,
    expected_hash: str,
    expected_target_path: str,
    expected_error: mlrun.errors.MLRunBaseError,
):
    if expected_error:
        with pytest.raises(expected_error):
            artifact.resolve_file_target_hash_path(
                source_path=artifact.spec.src_path, artifact_path=artifact_path
            )
        return
    file_hash, target_path = artifact.resolve_file_target_hash_path(
        source_path=artifact.spec.src_path, artifact_path=artifact_path
    )
    if not expected_hash:
        expected_hash = mlrun.utils.calculate_local_file_hash(artifact.spec.src_path)

    assert expected_hash == file_hash
    assert expected_hash in target_path

    if artifact.spec.format:
        assert target_path.endswith(f".{artifact.spec.format}")

    if expected_target_path:
        assert expected_target_path == target_path


@pytest.mark.parametrize(
    "artifact,artifact_path,expected_hash,expected_target_path,expected_error",
    [
        (
            mlrun.artifacts.Artifact("results", body="asdasdasdasdas"),
            "v3io://just/regular/path",
            "2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            "v3io://just/regular/path/2fc62a05b53733eb876e50f74b8fe35c809f05c3",
            None,
        ),
        (
            mlrun.artifacts.Artifact("results", body="asdasdasdasdas"),
            "v3io://just/regular/path",
            None,
            None,
            None,
        ),
        (
            mlrun.artifacts.Artifact("results", body=b"asdasdasdasdas"),
            "v3io://just/regular/path",
            None,
            None,
            None,
        ),
        (
            mlrun.artifacts.Artifact("results", body={"ba": "nana"}),
            "v3io://just/regular/path",
            None,
            None,
            TypeError,
        ),
        (
            mlrun.artifacts.Artifact("results", body="asdasdasdasdas"),
            None,
            None,
            None,
            mlrun.errors.MLRunInvalidArgumentError,
        ),
    ],
)
def test_resolve_body_hash_path(
    artifact: mlrun.artifacts.Artifact,
    artifact_path: str,
    expected_hash: str,
    expected_target_path: str,
    expected_error: typing.Union[mlrun.errors.MLRunBaseError, TypeError],
):
    if expected_error:
        with pytest.raises(expected_error):
            artifact.resolve_body_target_hash_path(
                body=artifact.get_body(), artifact_path=artifact_path
            )
        return
    body_hash, target_path = artifact.resolve_body_target_hash_path(
        body=artifact.get_body(), artifact_path=artifact_path
    )

    if not expected_hash:
        expected_hash = mlrun.artifacts.base.calculate_blob_hash(artifact.get_body())

    assert expected_hash == body_hash
    assert expected_hash in target_path

    if artifact.spec.format:
        assert target_path.endswith(f".{artifact.spec.format}")

    if expected_target_path:
        assert expected_target_path == target_path


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


def test_tag_not_in_model_spec():
    model_name = "my-model"
    tag = "some-tag"
    project_name = "model-spec-test"
    artifact_path = results_dir / project_name

    # create a project and log a model
    project = mlrun.new_project(project_name, save=False)
    project.log_model(
        model_name,
        body="model body",
        model_file="trained_model.pkl",
        tag=tag,
        artifact_path=artifact_path,
        upload=True,
    )

    # list the artifact path dir and verify the model spec file exists
    model_path = artifact_path / model_name
    files = os.listdir(model_path)
    assert mlrun.artifacts.model.model_spec_filename in files

    # open the model spec file and verify the tag is not there
    with open(model_path / mlrun.artifacts.model.model_spec_filename) as f:
        model_spec = yaml.load(f, Loader=yaml.FullLoader)

    assert "tag" not in model_spec, "tag should not be in model spec"
    assert "tag" not in model_spec["metadata"], "tag should not be in metadata"


def test_register_artifacts(rundb_mock):
    project_name = "my-projects"
    project = mlrun.new_project(project_name)
    artifact_key = "my-art"
    artifact_tag = "v1"
    project.set_artifact(
        artifact_key,
        artifact=mlrun.artifacts.Artifact(key=artifact_key, body=b"x=1"),
        tag=artifact_tag,
    )

    expected_tree = "my_uuid"
    with unittest.mock.patch.object(uuid, "uuid4", return_value=expected_tree):
        project.register_artifacts()

    artifact = project.get_artifact(artifact_key)
    assert artifact.tree == expected_tree
