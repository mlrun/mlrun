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
import os.path
import pathlib
import tempfile
import typing
import unittest.mock
import uuid
from contextlib import nullcontext as does_not_raise

import deepdiff
import pandas as pd
import pytest
import yaml

import mlrun
import mlrun.artifacts
from mlrun.artifacts.manager import extend_artifact_path
from mlrun.common.constants import MYSQL_MEDIUMBLOB_SIZE_BYTES
from mlrun.utils import StorePrefix
from tests import conftest

results_dir = (pathlib.Path(conftest.results) / "artifacts").absolute()


def test_artifacts_export_required_fields():
    artifact_classes = [
        mlrun.artifacts.Artifact,
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


@pytest.mark.parametrize(
    "target_path,upload_options",
    [
        ("s3://some/path", False),
        ("s3://some/path", True),
        ("s3://some/path", None),
        (None, False),
        (None, True),
        (None, None),
    ],
)
def test_log_artifact_with_target_path_and_upload_options(target_path, upload_options):
    artifact = mlrun.artifacts.Artifact(
        key="some-artifact", body="asdasdasdasdas", format="parquet"
    )
    with unittest.mock.patch.object(mlrun.datastore.DataItem, "put"):
        logged_artifact = mlrun.get_or_create_ctx("test").log_artifact(
            artifact, target_path=target_path, upload=upload_options
        )
    if upload_options or (not target_path and upload_options is None):
        # if upload is True or no target path was given and upload is None,
        # we expect the artifact to get uploaded and a target path to be generated
        assert logged_artifact.target_path is not None
        assert logged_artifact.metadata.hash is not None
    elif target_path:
        # if target path is given and upload is not True, we don't upload and therefore don't calculate hash
        assert logged_artifact.target_path == target_path
        assert logged_artifact.metadata.hash is None


@pytest.mark.parametrize(
    "artifact_key,expected",
    [
        ("artifact@key", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("artifact#key", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("artifact!key", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("artifact_key!", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("artifact/key", pytest.raises(mlrun.errors.MLRunInvalidArgumentError)),
        ("artifact_key123", does_not_raise()),
        ("artifact-key", does_not_raise()),
        ("artifact.key", does_not_raise()),
    ],
)
def test_log_artifact_with_invalid_key(artifact_key, expected):
    project = mlrun.new_project("test-project")
    target_path = "s3://some/path"
    artifact = mlrun.artifacts.Artifact(
        key=artifact_key, body="123", target_path=target_path
    )
    with expected:
        project.log_artifact(artifact)

    # now test log_artifact with db_key that is different than the artifact's key
    artifact = mlrun.artifacts.Artifact(
        key="some-key", body="123", target_path=target_path
    )
    artifact.spec.db_key = artifact_key
    with expected:
        project.log_artifact(artifact)

    # when storing an artifact with producer.kind="run", the value of db_key is modified to: producer.name + "_" + key
    # and in this case since key="some-key", the db_key will always be valid
    context = mlrun.get_or_create_ctx("test")
    try:
        context.log_artifact(item=artifact)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize(
    "local_path, fail",
    [
        ("s3://path/file.txt", False),
        ("", False),
        ("file://", False),
        ("file:///not_exists/file.txt", True),
        ("/not_exists/file.txt", True),
    ],
)
def test_ensure_artifact_source_file_exists(local_path, fail):
    artifact = mlrun.artifacts.Artifact(
        "artifact-name",
    )
    context = mlrun.get_or_create_ctx("test")
    if fail:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as error:
            context.log_artifact(item=artifact, local_path=local_path)
        assert "Failed to log an artifact, file does not exists" in str(error.value)
    else:
        if not local_path or local_path == "file://":
            df = pd.DataFrame({"num": [0, 1, 2], "color": ["green", "blue", "red"]})
            with tempfile.NamedTemporaryFile(suffix=".pq", delete=True) as temp_file:
                path = temp_file.name
                df.to_parquet(path)
                if local_path == "file://":
                    path = local_path + path
                context.log_artifact(item=artifact, local_path=path)
        else:
            context.log_artifact(item=artifact, local_path=local_path)


@pytest.mark.parametrize(
    "body_size,expectation",
    [
        (
            MYSQL_MEDIUMBLOB_SIZE_BYTES + 1,
            pytest.raises(mlrun.errors.MLRunBadRequestError),
        ),
        (MYSQL_MEDIUMBLOB_SIZE_BYTES - 1, does_not_raise()),
    ],
)
def test_ensure_fail_on_oversized_artifact(body_size, expectation):
    artifact = mlrun.artifacts.Artifact(
        "artifact-name",
        is_inline=True,
        body="a" * body_size,
    )
    context = mlrun.get_or_create_ctx("test")
    with expectation:
        context.log_artifact(item=artifact)


@pytest.mark.parametrize(
    "df, fail",
    [
        (pd.DataFrame({"num": [0, 1, 2], "color": ["green", "blue", "red"]}), False),
        (None, True),
    ],
)
def test_ensure_artifact_source_file_exists_by_df(df, fail):
    context = mlrun.get_or_create_ctx("test")

    with tempfile.TemporaryDirectory() as temp_dir:
        full_path = os.path.join(temp_dir, "df.parquet")
        if fail:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as error:
                context.log_dataset(key=str(uuid.uuid4()), df=df, local_path=full_path)
            assert "Failed to log an artifact, file does not exists" in str(error.value)
        else:
            context.log_dataset(key=str(uuid.uuid4()), df=df, local_path=full_path)


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


def test_producer_in_exported_artifact():
    project_name = "my-project"
    project = mlrun.new_project(project_name, save=False)

    artifact = project.log_artifact(
        "x", body="123", is_inline=True, artifact_path=results_dir
    )

    assert artifact.producer.get("kind") == "project"
    assert artifact.producer.get("name") == project_name

    artifact_path = f"{results_dir}/x.yaml"
    artifact.export(artifact_path)

    with open(artifact_path) as file:
        exported_artifact = yaml.load(file, Loader=yaml.FullLoader)
        assert "producer" in exported_artifact["spec"]
        assert exported_artifact["spec"]["producer"]["kind"] == "project"
        assert exported_artifact["spec"]["producer"]["name"] == project_name

    # remove the producer from the artifact and export it again
    artifact.producer = None
    artifact.export(artifact_path)

    with open(artifact_path) as file:
        exported_artifact = yaml.load(file, Loader=yaml.FullLoader)
        assert "producer" not in exported_artifact["spec"]


@pytest.mark.parametrize(
    "uri,expected_parsed_result",
    [
        # Full URI
        (
            "my-project/1234-1",
            ("my-project", "1234", "1"),
        ),
        # No iteration
        (
            "my-project/1234",
            ("my-project", "1234", ""),
        ),
        # No project
        (
            "1234-1",
            ("", "1234", "1"),
        ),
        # No UID
        (
            "my-project/-1",
            ("my-project", "", "1"),
        ),
        # just iteration
        (
            "-1",
            ("", "", "1"),
        ),
        # Nothing
        (
            "",
            ("", "", ""),
        ),
    ],
)
def test_artifact_producer_parse_uri(uri, expected_parsed_result):
    parsed_result = mlrun.artifacts.ArtifactProducer.parse_uri(uri)
    assert (
        deepdiff.DeepDiff(parsed_result, expected_parsed_result, ignore_order=True)
        == {}
    )
