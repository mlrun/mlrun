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

import os
import unittest.mock

import pytest

import mlrun.datastore
import mlrun.errors
import mlrun.utils.clones


@pytest.mark.parametrize(
    "ref,ref_type",
    [
        ("without-slash", "branch"),
        ("with/slash", "branch"),
        ("without-slash", "tag"),
        ("without/slash", "tag"),
    ],
)
def test_clone_git_refs(ref, ref_type):
    repo = "github.com/some-git-project/some-git-repo.git"
    url = f"git://{repo}#refs/{'heads' if ref_type == 'branch' else 'tags'}/{ref}"
    context = "non-existent-dir"
    branch = ref if ref_type == "branch" else None
    tag = ref if ref_type == "tag" else None

    with unittest.mock.patch("git.Repo.clone_from") as clone_from:
        _, repo_obj = mlrun.utils.clones.clone_git(url, context)
        clone_from.assert_called_once_with(
            f"https://{repo}", context, single_branch=True, b=branch
        )
        if tag:
            repo_obj.git.checkout.assert_called_once_with(tag)


@pytest.mark.parametrize(
    "url,secrets,enriched",
    [
        ("https://github.com/some-git-project", {"GIT_TOKEN": "123"}, True),
        ("https://github.com:8080/some-git-project", {"GIT_TOKEN": "123"}, True),
        ("https://github.com:8080/some-git-project", {}, False),
        ("git://somewhere:8080/else", {}, False),
    ],
)
def test_add_credentials_git_remote_url(url, secrets, enriched):
    resolved_url, url_enriched = mlrun.utils.clones.add_credentials_git_remote_url(
        url, secrets=secrets
    )
    if enriched:
        assert resolved_url.startswith("https://")
    else:
        assert url == resolved_url
    assert secrets.get("GIT_TOKEN", "") in resolved_url
    assert enriched is url_enriched


@pytest.mark.parametrize("project", [None, "my-project"])
def test_load_artifact_success(tmp_path, project):
    project_name = project or "my-project"
    source_uri = f"store://artifacts/{project_name}/handler.py"
    target_dir = str(tmp_path / "target")
    artifact_target_path = "s3://bucket/artifacts/handler.py"

    mock_artifact = unittest.mock.MagicMock()
    mock_artifact.get_target_path.return_value = artifact_target_path
    mock_artifact.spec.src_path = "handler.py"
    mock_dataitem = unittest.mock.MagicMock()

    with (
        unittest.mock.patch.object(mlrun.datastore, "is_store_uri", return_value=True),
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=mock_artifact
        ) as mock_get_resource,
        unittest.mock.patch(
            "mlrun.get_dataitem", return_value=mock_dataitem
        ) as mock_get_dataitem,
    ):
        result = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
            project=project,
        )

    expected_path = os.path.join(target_dir, "handler.py")

    # Assert returned path is under target_dir
    assert result == expected_path
    assert result.startswith(target_dir)

    # Assert directory was actually created
    assert os.path.isdir(target_dir)

    mock_get_resource.assert_called_once_with(source_uri, project=project)

    # Assert get_dataitem is called with the artifact's target path
    mock_get_dataitem.assert_called_once_with(artifact_target_path)

    # Assert download is called with the local destination path
    mock_dataitem.download.assert_called_once_with(expected_path)


@pytest.mark.parametrize(
    "source_uri,target_dir,is_store_uri_return,artifact_target_path,error_match",
    [
        # Missing source_uri
        ("", "/tmp/target", True, "s3://path", "source_uri is required"),
        # Missing target_dir
        (
            "store://artifacts/project/file.py",
            "",
            True,
            "s3://path",
            "target_dir is required",
        ),
        # Unsupported source type (not store://, git://, .zip, or .tar.gz)
        (
            "http://not-a-store/file.py",
            "/tmp/target",
            False,
            "s3://path",
            "Unsupported source type",
        ),
        # Artifact without target path
        (
            "store://artifacts/project/file.py",
            "/tmp/target",
            True,
            None,
            "does not have a valid target path",
        ),
    ],
)
def test_load_source_code_failures(
    source_uri, target_dir, is_store_uri_return, artifact_target_path, error_match
):
    # Test various failure scenarios for load_source_code
    mock_artifact = unittest.mock.MagicMock()
    mock_artifact.get_target_path.return_value = artifact_target_path

    with (
        unittest.mock.patch.object(
            mlrun.datastore, "is_store_uri", return_value=is_store_uri_return
        ),
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=mock_artifact
        ),
    ):
        with pytest.raises(ValueError, match=error_match):
            mlrun.utils.clones.load_source_code(
                source_uri=source_uri,
                target_dir=target_dir,
            )


def test_load_source_code_git(tmp_path):
    source_uri = "git://github.com/org/repo.git#main"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(mlrun.utils.clones, "clone_git") as mock_clone_git:
        result = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    assert result == target_dir
    mock_clone_git.assert_called_once_with(source_uri, target_dir)


def test_load_source_code_git_failure(tmp_path):
    source_uri = "git://github.com/org/repo.git"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(
        mlrun.utils.clones, "clone_git", side_effect=Exception("Clone failed")
    ):
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError, match="Failed to clone Git repository"
        ):
            mlrun.utils.clones.load_source_code(
                source_uri=source_uri,
                target_dir=target_dir,
            )


def test_load_source_code_zip(tmp_path):
    source_uri = "https://example.com/source.zip"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(mlrun.utils.clones, "clone_zip") as mock_clone_zip:
        result = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    assert result == target_dir
    mock_clone_zip.assert_called_once_with(source_uri, target_dir)


def test_load_source_code_tgz(tmp_path):
    source_uri = "https://example.com/source.tar.gz"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(mlrun.utils.clones, "clone_tgz") as mock_clone_tgz:
        result = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    assert result == target_dir
    mock_clone_tgz.assert_called_once_with(source_uri, target_dir)


def test_load_source_code_archive_failure(tmp_path):
    source_uri = "https://example.com/source.zip"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(
        mlrun.utils.clones, "clone_zip", side_effect=Exception("Extract failed")
    ):
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError, match="Failed to extract archive"
        ):
            mlrun.utils.clones.load_source_code(
                source_uri=source_uri,
                target_dir=target_dir,
            )
