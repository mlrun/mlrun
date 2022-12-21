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
import typing
import unittest.mock
from abc import ABCMeta
from contextlib import nullcontext as does_not_raise

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


@pytest.mark.parametrize(
    "artifact,expected_hash,expected_target_path,artifact_path,generate_target_path,tag,raises,is_logged_artifact",
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
    raises: typing.Union[ABCMeta, ValueError],
    is_logged_artifact: bool,
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

    with raises:
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
    if is_logged_artifact:
        if artifact.spec.format:
            assert logged_artifact.target_path.endswith(f".{artifact.spec.format}")

        if expected_target_path:
            assert expected_target_path == logged_artifact.target_path

        if expected_hash:
            assert expected_hash == logged_artifact.metadata.hash
            assert expected_hash in logged_artifact.target_path


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


def test_export_import():
    project = mlrun.new_project("log-mod", save=False)
    target_project = mlrun.new_project("log-mod2", save=False)
    for mode in [False, True]:
        mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = mode

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
            temp_path, model_spec, extra_dataitems = mlrun.artifacts.get_model(
                artifact.uri
            )
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
