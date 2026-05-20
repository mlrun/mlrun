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

import io
import os
import tarfile
import unittest.mock
import zipfile

import pytest

import mlrun
import mlrun.artifacts
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

    mock_artifact = unittest.mock.MagicMock(spec=mlrun.artifacts.Artifact)
    mock_artifact.kind = "artifact"
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
        returned_dir, returned_file_path = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
            project=project,
        )

    expected_local_file = os.path.join(target_dir, "handler.py")

    # store:// returns both the directory (workdir for runtime callers) and
    # the resolved file path (for callers that need to import a single file).
    assert returned_dir == target_dir
    assert returned_file_path == expected_local_file

    # Assert directory was actually created
    assert os.path.isdir(target_dir)

    mock_get_resource.assert_called_once_with(
        source_uri, project=project, secrets=None, data_store_secrets=None
    )

    # Assert get_dataitem is called with the artifact's target path
    mock_get_dataitem.assert_called_once_with(artifact_target_path, secrets=None)

    # Assert download is called with the local destination file path
    mock_dataitem.download.assert_called_once_with(expected_local_file)


def test_load_artifact_forwards_secrets(tmp_path):
    """Secrets passed to load_source_code are threaded to the resolver and downloader.

    Required so artifacts whose target path lives on a credential-protected
    store (S3/GCS/Azure) can still be fetched when the caller provides creds.
    """
    source_uri = "store://artifacts/project/handler.py"
    target_dir = str(tmp_path / "target")
    artifact_target_path = "s3://bucket/artifacts/handler.py"
    secrets = {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"}

    mock_artifact = unittest.mock.MagicMock(spec=mlrun.artifacts.Artifact)
    mock_artifact.kind = "artifact"
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
        mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
            secrets=secrets,
        )

    mock_get_resource.assert_called_once_with(
        source_uri,
        project=None,
        secrets=secrets,
        data_store_secrets=secrets,
    )
    mock_get_dataitem.assert_called_once_with(artifact_target_path, secrets=secrets)


def test_extract_source_store_uri_forwards_secrets():
    """extract_source forwards its secrets parameter through to load_source_code."""
    secrets = {"AWS_ACCESS_KEY_ID": "k"}
    with unittest.mock.patch("mlrun.utils.clones.load_source_code") as mock_load:
        mock_load.return_value = ("/tmp/workdir", "/tmp/workdir/my_func.py")
        mlrun.utils.clones.extract_source(
            source="store://artifacts/proj/my_func",
            workdir="/tmp/workdir",
            project="proj",
            secrets=secrets,
        )
        mock_load.assert_called_once_with(
            source_uri="store://artifacts/proj/my_func",
            target_dir="/tmp/workdir",
            project="proj",
            secrets=secrets,
        )


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
    mock_artifact = unittest.mock.MagicMock(spec=mlrun.artifacts.Artifact)
    mock_artifact.kind = "artifact"
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
        returned_dir, returned_file_path = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    # git/archive sources have no canonical entry file → file_path is None.
    assert returned_dir == target_dir
    assert returned_file_path is None
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
        returned_dir, returned_file_path = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    assert returned_dir == target_dir
    assert returned_file_path is None
    mock_clone_zip.assert_called_once_with(source_uri, target_dir)


def test_load_source_code_tgz(tmp_path):
    source_uri = "https://example.com/source.tar.gz"
    target_dir = str(tmp_path / "target")

    with unittest.mock.patch.object(mlrun.utils.clones, "clone_tgz") as mock_clone_tgz:
        returned_dir, returned_file_path = mlrun.utils.clones.load_source_code(
            source_uri=source_uri,
            target_dir=target_dir,
        )

    assert returned_dir == target_dir
    assert returned_file_path is None
    mock_clone_tgz.assert_called_once_with(source_uri, target_dir)


def test_extract_source_store_uri_delegates_to_load_source_code():
    """extract_source with store:// URI delegates to load_source_code and
    returns the directory (workdir for runtime callers)."""
    with unittest.mock.patch("mlrun.utils.clones.load_source_code") as mock_load:
        mock_load.return_value = ("/tmp/workdir", "/tmp/workdir/my_func.py")
        result = mlrun.utils.clones.extract_source(
            source="store://artifacts/proj/my_func",
            workdir="/tmp/workdir",
            project="proj",
        )
        mock_load.assert_called_once_with(
            source_uri="store://artifacts/proj/my_func",
            target_dir="/tmp/workdir",
            project="proj",
            secrets=None,
        )
        assert result == "/tmp/workdir"


def test_extract_source_store_uri_without_project():
    """extract_source with store:// but no project passes project=None."""
    with unittest.mock.patch("mlrun.utils.clones.load_source_code") as mock_load:
        mock_load.return_value = ("/tmp/workdir", "/tmp/workdir/my_func.py")
        mlrun.utils.clones.extract_source(
            source="store://artifacts/proj/my_func",
            workdir="/tmp/workdir",
        )
        mock_load.assert_called_once_with(
            source_uri="store://artifacts/proj/my_func",
            target_dir="/tmp/workdir",
            project=None,
            secrets=None,
        )


def test_extract_source_store_uri_default_workdir():
    """extract_source with store:// and no workdir uses default ./code dir."""
    with unittest.mock.patch("mlrun.utils.clones.load_source_code") as mock_load:
        expected_target = os.path.realpath("./code")
        mock_load.return_value = (expected_target, f"{expected_target}/my_func.py")
        mlrun.utils.clones.extract_source(
            source="store://artifacts/proj/my_func",
            project="proj",
        )
        mock_load.assert_called_once_with(
            source_uri="store://artifacts/proj/my_func",
            target_dir=expected_target,
            project="proj",
            secrets=None,
        )


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


@pytest.mark.parametrize(
    "src_path, target_path, key, expected",
    [
        # src_path wins when set
        ("foo.py", "s3://bucket/abcdef", "my-key", "foo.py"),
        # src_path wins even when it's a relative path with ./
        ("./sub/foo.py", "s3://bucket/abcdef", "my-key", "foo.py"),
        # src_path with parent traversal -> basename strips it
        ("../escape.py", "s3://bucket/abcdef", "my-key", "escape.py"),
        # src_path empty -> fall back to target_path
        ("", "s3://bucket/handler.py", "my-key", "handler.py"),
        # both empty -> fall back to key
        ("", "", "my-key", "my-key"),
        # src_path None -> fall back chain
        (None, None, "key/with/slashes", "slashes"),
        # all three fields empty / None -> ""
        ("", "", "", ""),
        (None, None, None, ""),
        # all three basename to empty (path-only) -> ""
        ("dir/", "s3://bucket/", "key/", ""),
    ],
)
def test_resolve_artifact_filename(src_path, target_path, key, expected):
    artifact = unittest.mock.MagicMock()
    artifact.spec.src_path = src_path
    artifact.spec.target_path = target_path
    artifact.metadata.key = key

    result = mlrun.utils.clones.resolve_artifact_filename(artifact)

    assert result == expected


@pytest.mark.parametrize(
    "body, expected_bytes",
    [
        (b"binary content\n", b"binary content\n"),
        ("text content\n", b"text content\n"),
        (b"", b""),
        ("", b""),
    ],
)
def test_write_body_to_path(tmp_path, body, expected_bytes):
    target = tmp_path / "out.bin"

    mlrun.utils.clones._write_body_to_path(body, str(target))

    assert target.read_bytes() == expected_bytes


def test_write_body_to_path_rejects_non_str_bytes(tmp_path):
    """A body that is neither str nor bytes must raise rather than silently
    writing a corrupted file (e.g. bytes(int) producing a zero buffer)."""
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match="Unsupported artifact body type"
    ):
        mlrun.utils.clones._write_body_to_path(123, str(tmp_path / "out.bin"))


# ---------------------------------------------------------------------------
# Archive builder helpers
# ---------------------------------------------------------------------------


def _make_zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _make_tgz_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _open_zip(path):
    return zipfile.ZipFile(path)


def _open_tar(path):
    return tarfile.open(path, "r:gz")


# Per-format parameter bundles reused across the archive tests below.
# Each tuple supplies everything that varies between zip and tar.gz so the
# tests themselves only express the behavior under test.
ARCHIVE_FORMATS = [
    pytest.param(
        _make_zip_bytes,
        ".zip",
        _open_zip,
        "_safe_extract_zip",
        zipfile.BadZipFile,
        id="zip",
    ),
    pytest.param(
        _make_tgz_bytes,
        ".tar.gz",
        _open_tar,
        "_safe_extract_tar",
        tarfile.ReadError,
        id="tgz",
    ),
]


def _mock_code_artifact(
    *,
    body=None,
    src_path="",
    target_path="",
    key="",
    kind=None,
    uri="store://artifacts/proj/key",
):
    """Build a MagicMock standing in for a resolved code artifact."""
    artifact = unittest.mock.MagicMock(spec=mlrun.artifacts.CodeArtifact)
    artifact.kind = mlrun.artifacts.CodeArtifact.kind if kind is None else kind
    artifact.spec.get_body.return_value = body
    artifact.spec.src_path = src_path
    artifact.spec.target_path = target_path
    artifact.metadata.key = key
    artifact.uri = uri
    return artifact


# ---------------------------------------------------------------------------
# Zip-slip-safe extractor tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_archive, suffix, opener, extract_fn_name, _bad_archive_exc", ARCHIVE_FORMATS
)
def test_safe_extract_archive_extracts_safe_members(
    tmp_path, make_archive, suffix, opener, extract_fn_name, _bad_archive_exc
):
    members = {"a.py": b"print('a')\n", "pkg/b.py": b"print('b')\n"}
    archive_path = tmp_path / f"good{suffix}"
    archive_path.write_bytes(make_archive(members))
    extract_fn = getattr(mlrun.utils.clones, extract_fn_name)

    with opener(str(archive_path)) as archive:
        extract_fn(archive, str(tmp_path))

    assert (tmp_path / "a.py").read_bytes() == members["a.py"]
    assert (tmp_path / "pkg" / "b.py").read_bytes() == members["pkg/b.py"]


@pytest.mark.parametrize(
    "make_archive, suffix, opener, extract_fn_name, _bad_archive_exc", ARCHIVE_FORMATS
)
def test_safe_extract_archive_rejects_path_traversal(
    tmp_path, make_archive, suffix, opener, extract_fn_name, _bad_archive_exc
):
    archive_path = tmp_path / f"evil{suffix}"
    archive_path.write_bytes(make_archive({"../escape.py": b"print('escape')\n"}))
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    extract_fn = getattr(mlrun.utils.clones, extract_fn_name)

    with opener(str(archive_path)) as archive:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="escape"):
            extract_fn(archive, str(target_dir))

    # No file leaked outside target_dir
    assert not (tmp_path / "escape.py").exists()


def test_safe_extract_tar_rejects_linkname_traversal(tmp_path):
    """A symlink whose linkname escapes target_dir must not be extracted."""
    archive_path = tmp_path / "evil_link.tar.gz"
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with tarfile.open(archive_path, "w:gz") as tf:
        info = tarfile.TarInfo(name="benign-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../etc/passwd"
        tf.addfile(info)

    # tarfile's PEP 706 data filter raises LinkOutsideDestinationError
    # for symlinks whose linkname escapes the destination.
    with tarfile.open(archive_path, "r:gz") as tf:
        with pytest.raises(tarfile.LinkOutsideDestinationError):
            mlrun.utils.clones._safe_extract_tar(tf, str(target_dir))

    # Nothing landed on disk under target_dir.
    assert list(target_dir.iterdir()) == []


@pytest.mark.parametrize(
    "make_archive, suffix, _opener, _extract_fn_name, _bad_archive_exc",
    ARCHIVE_FORMATS,
)
def test_maybe_extract_archive(
    tmp_path, make_archive, suffix, _opener, _extract_fn_name, _bad_archive_exc
):
    members = {"a.py": b"a", "b.py": b"b"}
    archive = tmp_path / f"src{suffix}"
    archive.write_bytes(make_archive(members))

    extracted = mlrun.utils.clones._maybe_extract_archive(str(archive), str(tmp_path))

    assert extracted is True
    # Removal is the caller's responsibility; the archive stays on disk here.
    assert archive.exists()
    assert (tmp_path / "a.py").read_bytes() == members["a.py"]
    assert (tmp_path / "b.py").read_bytes() == members["b.py"]


def test_maybe_extract_archive_tgz_suffix(tmp_path):
    """`.tgz` is gzipped tar and must be extracted like `.tar.gz`."""
    members = {"a.py": b"a", "b.py": b"b"}
    archive = tmp_path / "src.tgz"
    archive.write_bytes(_make_tgz_bytes(members))

    extracted = mlrun.utils.clones._maybe_extract_archive(str(archive), str(tmp_path))

    assert extracted is True
    assert (tmp_path / "a.py").read_bytes() == members["a.py"]
    assert (tmp_path / "b.py").read_bytes() == members["b.py"]


def test_maybe_extract_archive_non_archive_is_noop(tmp_path):
    payload = b"print('hello')\n"
    file_path = tmp_path / "hello.py"
    file_path.write_bytes(payload)

    mlrun.utils.clones._maybe_extract_archive(str(file_path), str(tmp_path))
    assert file_path.read_bytes() == payload


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"print('inline body')\n", id="non-empty"),
        # Pins that None vs b"" is the discriminator on the body branch,
        # not a generic truthiness check.
        pytest.param(b"", id="empty"),
    ],
)
def test_load_code_artifact_body_only_single_file(tmp_path, body):
    artifact = _mock_code_artifact(body=body, src_path="hello.py", key="hello")
    target_dir = str(tmp_path)

    with unittest.mock.patch("mlrun.get_dataitem") as mock_get_dataitem:
        result = mlrun.utils.clones._load_code_artifact(
            artifact, target_dir, secrets=None
        )

    assert result == (target_dir, os.path.join(target_dir, "hello.py"))
    assert (tmp_path / "hello.py").read_bytes() == body
    mock_get_dataitem.assert_not_called()


def test_load_code_artifact_target_path_single_file(tmp_path):
    artifact = _mock_code_artifact(src_path="handler.py")
    artifact.get_target_path.return_value = "s3://bucket/abc/handler.py"
    target_dir = str(tmp_path)

    def fake_download(dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(b"print('handler')\n")

    mock_dataitem = unittest.mock.MagicMock()
    mock_dataitem.download.side_effect = fake_download
    with unittest.mock.patch(
        "mlrun.get_dataitem", return_value=mock_dataitem
    ) as mock_get_dataitem:
        result = mlrun.utils.clones._load_code_artifact(
            artifact, target_dir, secrets=None
        )

    # A plain source file is content-detected as a non-archive and left as-is.
    assert result == (target_dir, os.path.join(target_dir, "handler.py"))
    assert (tmp_path / "handler.py").read_bytes() == b"print('handler')\n"
    mock_get_dataitem.assert_called_once_with(
        "s3://bucket/abc/handler.py", secrets=None
    )
    mock_dataitem.download.assert_called_once_with(
        os.path.join(target_dir, "handler.py")
    )


@pytest.mark.parametrize(
    "make_archive, suffix, _opener, _extract_fn_name, _bad_archive_exc",
    ARCHIVE_FORMATS,
)
def test_load_code_artifact_target_path_archive_extracted(
    tmp_path, make_archive, suffix, _opener, _extract_fn_name, _bad_archive_exc
):
    members = {"a.py": b"print('a')\n", "b.py": b"print('b')\n"}
    archive_bytes = make_archive(members)
    artifact = _mock_code_artifact(src_path=f"src{suffix}")
    artifact.get_target_path.return_value = f"s3://bucket/abc/src{suffix}"
    target_dir = str(tmp_path)

    def fake_download(dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(archive_bytes)

    mock_dataitem = unittest.mock.MagicMock()
    mock_dataitem.download.side_effect = fake_download

    with unittest.mock.patch("mlrun.get_dataitem", return_value=mock_dataitem):
        result = mlrun.utils.clones._load_code_artifact(
            artifact, target_dir, secrets=None
        )

    assert result == (target_dir, None)
    assert not (tmp_path / f"src{suffix}").exists()
    assert (tmp_path / "a.py").read_bytes() == members["a.py"]
    assert (tmp_path / "b.py").read_bytes() == members["b.py"]


def test_load_code_artifact_no_body_no_target_path_raises(tmp_path):
    artifact = _mock_code_artifact(src_path="handler.py")
    artifact.get_target_path.return_value = ""

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="neither inline body nor a target_path",
    ):
        mlrun.utils.clones._load_code_artifact(artifact, str(tmp_path), secrets=None)


def test_load_code_artifact_body_with_unresolvable_filename_raises(tmp_path):
    """Body-backed code artifact with no resolvable filename anywhere
    (src_path / target_path / metadata.key all empty or basename-empty)
    must raise typed MLRunInvalidArgumentError. Without this guard, the
    join produces target_dir itself, and the body write crashes with
    IsADirectoryError instead of a typed error.
    """
    artifact = _mock_code_artifact(body=b"hi", src_path="", target_path="", key="")

    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="has no resolvable filename",
    ):
        mlrun.utils.clones._load_code_artifact(artifact, str(tmp_path), secrets=None)


def test_load_code_artifact_target_path_archive_extension_not_an_archive(tmp_path):
    """Archive detection is by content, not suffix: a payload named like an
    archive but whose bytes are not a valid zip/tar is left as a single file
    (not extracted), and the returned path is the file itself.
    """
    artifact = _mock_code_artifact(src_path="broken.zip")
    artifact.get_target_path.return_value = "s3://bucket/abc/broken.zip"
    target_dir = str(tmp_path)

    def fake_download(dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(b"not actually an archive")

    mock_dataitem = unittest.mock.MagicMock()
    mock_dataitem.download.side_effect = fake_download

    with unittest.mock.patch("mlrun.get_dataitem", return_value=mock_dataitem):
        result = mlrun.utils.clones._load_code_artifact(
            artifact, target_dir, secrets=None
        )

    assert result == (target_dir, os.path.join(target_dir, "broken.zip"))
    assert (tmp_path / "broken.zip").read_bytes() == b"not actually an archive"


# ---------------------------------------------------------------------------
# Kind switch in _load_store_artifact
# ---------------------------------------------------------------------------


def test_load_store_artifact_none_resolution_raises(tmp_path):
    """get_store_resource can return None (e.g. a link artifact whose target
    re-read is falsy); the resolver must surface a typed NotFound rather than
    an AttributeError on None.kind."""
    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=None
        ),
        pytest.raises(
            mlrun.errors.MLRunNotFoundError, match="did not resolve to an artifact"
        ),
    ):
        mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/missing", str(tmp_path)
        )


def test_load_store_artifact_non_artifact_resolution_raises(tmp_path):
    """get_store_resource can return FeatureSet / DataItem for non-artifact
    store URIs; the resolver must reject those before touching .kind."""
    not_an_artifact = unittest.mock.MagicMock()

    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=not_an_artifact
        ),
        pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match="expected an artifact"
        ),
    ):
        mlrun.utils.clones._load_store_artifact(
            "store://feature-sets/proj/x", str(tmp_path)
        )


def test_load_store_artifact_code_kind_uses_body(tmp_path):
    body = b"print('code artifact')\n"
    artifact = _mock_code_artifact(body=body, src_path="code.py", key="code")
    target_dir = str(tmp_path)

    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=artifact
        ),
        unittest.mock.patch("mlrun.get_dataitem") as mock_get_dataitem,
    ):
        result = mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/code", target_dir
        )

    assert result == (target_dir, os.path.join(target_dir, "code.py"))
    assert (tmp_path / "code.py").read_bytes() == body
    mock_get_dataitem.assert_not_called()


def test_load_store_artifact_non_code_kind_skips_body(tmp_path):
    """A non-code artifact with body present must NOT be read; today's
    target-path download is the only path."""
    # Body present but should be ignored by the non-code branch.
    artifact = _mock_code_artifact(
        body=b"would-be-body", src_path="model.bin", kind="model"
    )
    artifact.get_target_path.return_value = "s3://bucket/model.bin"
    target_dir = str(tmp_path)

    mock_dataitem = unittest.mock.MagicMock()
    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=artifact
        ),
        unittest.mock.patch(
            "mlrun.get_dataitem", return_value=mock_dataitem
        ) as mock_get_dataitem,
    ):
        result = mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/model", target_dir
        )

    assert result == (target_dir, os.path.join(target_dir, "model.bin"))
    mock_get_dataitem.assert_called_once_with("s3://bucket/model.bin", secrets=None)
    mock_dataitem.download.assert_called_once()
    # Body was NOT consulted (the spec's get_body mock would still be
    # callable, but the production path must not have called it).
    artifact.spec.get_body.assert_not_called()


def test_load_store_artifact_non_code_zip_not_extracted(tmp_path):
    """A non-code artifact whose target is a .zip must NOT be extracted;
    extraction is a code-artifact-only behavior."""
    archive_bytes = _make_zip_bytes({"a.py": b"a"})

    artifact = _mock_code_artifact(src_path="bundle.zip", kind="model")
    artifact.get_target_path.return_value = "s3://bucket/bundle.zip"
    target_dir = str(tmp_path)

    def fake_download(dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(archive_bytes)

    mock_dataitem = unittest.mock.MagicMock()
    mock_dataitem.download.side_effect = fake_download

    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=artifact
        ),
        unittest.mock.patch("mlrun.get_dataitem", return_value=mock_dataitem),
    ):
        mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/bundle", target_dir
        )

    # The archive is still on disk, NOT extracted.
    assert (tmp_path / "bundle.zip").read_bytes() == archive_bytes
    assert not (tmp_path / "a.py").exists()


def test_load_store_artifact_non_code_empty_src_path_falls_back_to_target_basename(
    tmp_path,
):
    """When src_path is unset, the on-disk filename falls back to the
    target_path basename. Pins the historical _load_store_artifact
    behavior now relocated into _download_artifact_to_dir.
    """
    artifact = _mock_code_artifact(
        src_path="", target_path="s3://bucket/dir/payload.bin", kind="model"
    )
    artifact.get_target_path.return_value = "s3://bucket/dir/payload.bin"
    target_dir = str(tmp_path)

    mock_dataitem = unittest.mock.MagicMock()
    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=artifact
        ),
        unittest.mock.patch(
            "mlrun.get_dataitem", return_value=mock_dataitem
        ) as mock_get_dataitem,
    ):
        result = mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/payload", target_dir
        )

    assert result == (target_dir, os.path.join(target_dir, "payload.bin"))
    mock_get_dataitem.assert_called_once_with(
        "s3://bucket/dir/payload.bin", secrets=None
    )


def test_download_artifact_to_dir_unresolvable_filename_raises(tmp_path):
    """target_path is truthy (passes the early check) but its basename is
    empty and no other field provides a filename; the explicit
    ``has no resolvable filename`` raise must fire before any download.
    """
    artifact = _mock_code_artifact(
        src_path="", target_path="s3://bucket/", key="", kind="model"
    )
    artifact.get_target_path.return_value = "s3://bucket/"

    with (
        unittest.mock.patch.object(
            mlrun.datastore, "get_store_resource", return_value=artifact
        ),
        unittest.mock.patch("mlrun.get_dataitem") as mock_get_dataitem,
        pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="has no resolvable filename",
        ),
    ):
        mlrun.utils.clones._load_store_artifact(
            "store://artifacts/proj/no-name", str(tmp_path)
        )

    mock_get_dataitem.assert_not_called()
