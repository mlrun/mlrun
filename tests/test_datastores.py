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
import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd
import pytest

import mlrun
import mlrun.errors
from mlrun.artifacts import ModelArtifact
from mlrun.artifacts.base import LegacyLinkArtifact, LinkArtifact
from mlrun.artifacts.model import LegacyModelArtifact
from tests.conftest import rundb_path

mlrun.mlconf.dbpath = rundb_path

raw_data = {
    "name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    "age": [42, 52, 36, 24, 73],
}
df = pd.DataFrame(raw_data, columns=["name", "age"])


def test_in_memory():
    context = mlrun.get_or_create_ctx("test-in-mem")
    context.artifact_path = "memory://"
    k1 = context.log_artifact("k1", body="abc")
    k2 = context.log_dataset("k2", df=df)

    data = mlrun.datastore.set_in_memory_item("aa", "123")
    in_memory_store = mlrun.datastore.get_in_memory_items()
    new_df = mlrun.run.get_dataitem(k2.get_target_path()).as_df()

    assert len(in_memory_store) == 3, "data not written properly to in mem store"
    assert data.get() == "123", "in mem store failed to get/put"
    assert len(new_df) == 5, "in mem store failed dataframe test"
    assert (
        mlrun.run.get_dataitem(k1.get_target_path()).get() == "abc"
    ), "failed to log in mem artifact"


def test_file():
    with TemporaryDirectory() as tmpdir:
        print(tmpdir)

        data = mlrun.run.get_dataitem(tmpdir + "/test1.txt")
        data.put("abc")
        assert data.get() == b"abc", "failed put/get test"
        assert data.stat().size == 3, "got wrong file size"
        print(data.stat())

        context = mlrun.get_or_create_ctx("test-file")
        context.artifact_path = tmpdir
        k1 = context.log_artifact("k1", body="abc", local_path="x.txt")
        k2 = context.log_dataset("k2", df=df, format="csv", db_key="k2key")
        print("k2 url:", k2.uri)

        # test that we can get the artifact as dataitem
        assert k1.to_dataitem().get(encoding="utf-8") == "abc", "wrong .dataitem result"

        assert "test1.txt" in mlrun.run.get_dataitem(tmpdir).listdir(), "failed listdir"

        expected = [f"{tmpdir}/test1.txt", k2.get_target_path(), k1.get_target_path()]
        for a in expected:
            assert os.path.isfile(a) and a.startswith(
                tmpdir
            ), f"artifact {a} was not generated"

        new_fd = mlrun.run.get_dataitem(k2.get_target_path()).as_df()

        assert len(new_fd) == 5, "failed dataframe test"
        assert (
            mlrun.run.get_dataitem(k1.get_target_path()).get() == b"abc"
        ), "failed to log in file artifact"

        name = k2.uri
        artifact, _ = mlrun.artifacts.get_artifact_meta(name)
        print(artifact.to_yaml())
        mlrun.artifacts.update_dataset_meta(
            artifact, extra_data={"k1": k1}, column_metadata={"age": "great"}
        )
        artifact, _ = mlrun.artifacts.get_artifact_meta(name)
        print(artifact.to_yaml())
        assert artifact.column_metadata == {
            "age": "great"
        }, "failed artifact update test"


def test_parse_url_preserve_case():
    url = "store://Hedi/mlrun-dbd7ef-training_mymodel#a5dc8e34a46240bb9a07cd9deb3609c7"
    expected_endpoint = "Hedi"
    _, endpoint, _ = mlrun.datastore.datastore.parse_url(url)
    assert expected_endpoint, endpoint


def test_get_store_artifact_url_parsing():
    db = Mock()
    cases = [
        {
            "url": "store:///artifact_key",
            "project": "default",
            "key": "artifact_key",
            "tag": None,
            "iter": None,
        },
        {
            "url": "store://project_name/artifact_key",
            "project": "project_name",
            "key": "artifact_key",
            "tag": None,
            "iter": None,
        },
        {
            "url": "store://Project_Name/Artifact_Key@ABC",
            "project": "Project_Name",
            "key": "Artifact_Key",
            "tag": "ABC",
            "iter": None,
        },
        {
            "url": "store://project_name/artifact_key@a5dc8e34a46240bb9a07cd9deb3609c7",
            "project": "project_name",
            "key": "artifact_key",
            "tag": "a5dc8e34a46240bb9a07cd9deb3609c7",
            "iter": None,
        },
        {
            "url": "store://project_name/artifact_key#1",
            "project": "project_name",
            "key": "artifact_key",
            "tag": None,
            "iter": 1,
        },
        {
            "url": "store://project_name/artifact_key:latest",
            "project": "project_name",
            "key": "artifact_key",
            "tag": "latest",
            "iter": None,
        },
        {
            "url": "store:///ArtifacT_key#1:some_Tag",
            "project": "default",
            "key": "ArtifacT_key",
            "tag": "some_Tag",
            "iter": 1,
        },
        {
            "url": "store:///ArtifacT_key#1@Some_Tag",
            "project": "default",
            "key": "ArtifacT_key",
            "tag": "Some_Tag",
            "iter": 1,
        },
        {
            "url": "store://Project_Name/Artifact_Key:ABC",
            "project": "Project_Name",
            "key": "Artifact_Key",
            "tag": "ABC",
            "iter": None,
        },
    ]
    for case in cases:
        url = case["url"]
        expected_project = case["project"]
        expected_key = case["key"]
        expected_tag = case["tag"]
        expected_iter = case["iter"]

        def mock_read_artifact(key, tag=None, iter=None, project=""):
            assert expected_project == project
            assert expected_key == key
            assert expected_tag == tag
            assert expected_iter == iter
            return {}

        db.read_artifact = mock_read_artifact
        mlrun.datastore.store_resources.get_store_resource(url, db)


@pytest.mark.parametrize("legacy_format", [False, True])
def test_get_store_resource_with_linked_artifacts(legacy_format):
    artifact_key = "key1"
    project = "test_project"
    link_iteration = 7

    if legacy_format:
        link_artifact = LegacyLinkArtifact(
            key=artifact_key, target_path="/some/path", link_iteration=link_iteration
        )
        link_artifact.project = project
        model_artifact = LegacyModelArtifact(
            key=f"{artifact_key}#{link_iteration}",
            target_path="/some/path/again",
            body="just a body",
        )
        model_artifact.project = project
    else:
        link_artifact = LinkArtifact(
            key=artifact_key,
            project=project,
            target_path="/some/path",
            link_iteration=link_iteration,
        )
        model_artifact = ModelArtifact(
            key=f"{artifact_key}#{link_iteration}",
            project=project,
            target_path="/some/path/again",
            body="just a body",
        )

    mock_artifacts = [link_artifact, model_artifact]

    def mock_read_artifact(key, tag=None, iter=None, project=""):
        for artifact in mock_artifacts:
            key_ = f"{key}#{iter}" if iter else key
            if artifact.key == key_:
                return artifact.to_dict()
        return {}

    db = Mock()
    db.read_artifact = mock_read_artifact

    url = f"store://{project}/{artifact_key}"
    result = mlrun.datastore.store_resources.get_store_resource(url, db)
    assert result.kind == "model" and result.key == f"{artifact_key}#{link_iteration}"


@pytest.mark.usefixtures("patch_file_forbidden")
def test_forbidden_file_access():
    store = mlrun.datastore.datastore.StoreManager(
        secrets={"V3IO_ACCESS_KEY": "some-access-key"}
    )

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        obj = store.object("v3io://some-system/some-dir/")
        obj.listdir()

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        obj = store.object("v3io://some-system/some-dir/some-file")
        obj.get()

    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        obj = store.object("v3io://some-system/some-dir/some-file")
        obj.stat()


def test_verify_data_stores_are_not_cached_in_api_when_not_needed():
    mlrun.config._is_running_as_api = True

    user1_secrets = {"V3IO_ACCESS_KEY": "user1-access-key"}
    user1_objpath = "v3io://some-system/some-dir/user1"

    user2_secrets = {"V3IO_ACCESS_KEY": "user2-access-key"}
    user2_objpath = "v3io://some-system/some-dir/user2"

    user3_objpath = "v3io://some-system/some-dir/user3"
    store = mlrun.datastore.datastore.StoreManager(
        secrets={"V3IO_ACCESS_KEY": "api-access-key"}
    )
    obj = store.object(url=user1_objpath, secrets=user1_secrets)
    assert store._stores == {}
    assert obj._store._secrets == user1_secrets

    obj2 = store.object(url=user2_objpath, secrets=user2_secrets)
    assert store._stores == {}
    assert obj2._store._secrets == user2_secrets

    obj3 = store.object(url=user3_objpath)
    assert store._stores == {}
    assert obj3._store._secrets == {}


def test_verify_data_stores_are_cached_when_not_api():
    user1_secrets = {"V3IO_ACCESS_KEY": "user1-access-key"}
    user1_objpath = "v3io://some-system/some-dir/user1"

    user2_secrets = {"V3IO_ACCESS_KEY": "user2-access-key"}
    user2_objpath = "v3io://some-system/some-dir/user2"

    user3_objpath = "v3io://some-system/some-dir/user3"
    store = mlrun.datastore.datastore.StoreManager(
        secrets={"V3IO_ACCESS_KEY": "api-access-key"}
    )
    # if secrets provided then store is not cached
    obj = store.object(url=user1_objpath, secrets=user1_secrets)
    assert store._stores == {}
    assert obj._store._secrets == user1_secrets

    # if secrets provided then store is not cached
    obj2 = store.object(url=user2_objpath, secrets=user2_secrets)
    assert store._stores == {}
    assert obj2._store._secrets == user2_secrets

    # if no secrets provided then store is cached
    obj3 = store.object(url=user3_objpath)
    assert len(store._stores) == 1
    assert store._stores["v3io://some-system"] is not None
    assert obj3._store._secrets == {}

    # if secrets provided then store is not cached
    obj2 = store.object(url=user2_objpath, secrets=user2_secrets)
    assert len(store._stores) == 1
    assert obj2._store._secrets == user2_secrets
    # the store is not cached so the secrets are not updated, because this is the same store type as the one cached,
    # so we verify that the secrets are not updated
    assert store._stores["v3io://some-system"]._secrets == {}


def test_fsspec():
    with TemporaryDirectory() as tmpdir:
        print(tmpdir)
        store, _ = mlrun.store_manager.get_or_create_store(tmpdir)
        fs = store.get_filesystem(False)
        with store.open(tmpdir + "/1x.txt", "w") as fp:
            fp.write("123")
        with mlrun.get_dataitem(tmpdir + "/2x.txt").open("w") as fp:
            fp.write("456")
        files = fs.ls(tmpdir)
        assert len(files) == 2, "2 test files were not written"
        assert files[0].endswith("x.txt"), "wrong file name"
        assert fs.open(tmpdir + "/1x.txt", "r").read() == "123", "wrong file content"


@pytest.mark.parametrize("virtual_path", ["/dummy/path", "c:\\dummy\\path"])
def test_virtual_to_real_map(virtual_path):
    # test that the virtual dir (/dummy/path) is replaced with a real dir
    virtual_to_real_map = mlrun.mlconf.storage.virtual_to_real_path

    with TemporaryDirectory() as tmpdir:
        print(tmpdir)
        mlrun.mlconf.storage.virtual_to_real_path = f"{virtual_path}::{tmpdir}"

        data = mlrun.run.get_dataitem(f"{virtual_path}/test1.txt")
        data.put("abc")
        assert data.get() == b"abc", "failed put/get test"
        assert data.stat().size == 3, "got wrong file size"
        assert os.path.isfile(os.path.join(tmpdir, "test1.txt"))

    mlrun.mlconf.storage.virtual_to_real_path = virtual_to_real_map
