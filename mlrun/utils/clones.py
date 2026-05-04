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
import shutil
import tarfile
import tempfile
import zipfile
from os import path, remove
from urllib.parse import urlparse

from git import Repo

import mlrun

from .helpers import is_store_uri, logger


def _remove_directory_contents(target_dir):
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def _prep_dir(source, target_dir, suffix, secrets, clone):
    if not target_dir:
        raise ValueError("please specify a target (context) directory for clone")
    if clone and path.exists(target_dir) and path.isdir(target_dir):
        _remove_directory_contents(target_dir)

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    mlrun.get_dataitem(source, secrets).download(temp_file)
    return temp_file


def get_git_username_password_from_token(token):
    # Github's access tokens have a known prefix according to their type. See
    # https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#githubs-token-formats
    # We distinguish new fine-grained access tokens (begin with "github_pat_" from classic tokens.
    if token.startswith("github_pat_") or token.startswith("glpat"):
        username = "oauth2"
        password = token
    else:
        username = token
        password = "x-oauth-basic"
    return username, password


def clone_zip(source, target_dir, secrets=None, clone=True):
    tmpfile = _prep_dir(source, target_dir, ".zip", secrets, clone)
    with zipfile.ZipFile(tmpfile, "r") as zf:
        zf.extractall(target_dir)
    remove(tmpfile)  # delete zipped file


def clone_tgz(source, target_dir, secrets=None, clone=True):
    tmpfile = _prep_dir(source, target_dir, ".tar.gz", secrets, clone)
    with tarfile.TarFile.open(tmpfile, "r:*") as tf:
        tf.extractall(path=target_dir)
    remove(tmpfile)  # delete zipped file


def get_repo_url(repo):
    url = ""
    remotes = [remote.url for remote in repo.remotes]
    if not remotes:
        return ""

    url = remotes[0]
    url = url.replace("https://", "git://")
    try:
        url = f"{url}#refs/heads/{repo.active_branch.name}"
    except Exception:
        pass

    return url


def add_credentials_git_remote_url(url: str, secrets=None) -> tuple[str, bool]:
    """Enrich a Git remote URL with credential related secrets, if any are available
    If no secrets are supplied, or if the secrets are insufficient, the original URL is returned
    Besides the URL, this function also returns a bool indicating if any enrichment was done

    :param url:     git remote URL to be enriched
    :param secrets: dict or SecretsStore with Git credentials e.g. secrets={"GIT_TOKEN": token}

    :returns: tuple with the final URL and a boolean indicating if any enrichment was done
    """

    def get_secret(key):
        return mlrun.get_secret_or_env(key, secret_provider=secrets)

    url_obj = urlparse(url)

    username = url_obj.username or get_secret("GIT_USERNAME") or get_secret("git_user")
    password = (
        url_obj.password
        or get_secret("GIT_PASSWORD")
        or get_secret("git_password")
        or ""
    )
    token = (
        get_secret("GITHUB_TOKEN")
        or get_secret("GITLAB_TOKEN")
        or get_secret("GIT_TOKEN")
    )
    if token:
        username, password = get_git_username_password_from_token(token)

    if username:
        return f"https://{username}:{password}@{url_obj.netloc}{url_obj.path}", True
    return url, False


def clone_git(url: str, context: str, secrets=None, clone: bool = True):
    """Clone a remote Git repository in the local context

    :param url:     git remote URL
    :param context: local directory in which the repository must be stored
    :param secrets: dict or SecretsStore with Git credentials e.g. secrets={"GIT_TOKEN": token}
    :param clone:   delete all files and folders in "context" if there are any
    """

    url_obj = urlparse(url)
    if not context:
        raise ValueError("please specify a target (context) directory for clone")

    if path.exists(context) and path.isdir(context):
        if clone:
            _remove_directory_contents(context)
        else:
            if os.path.exists(context) and len(os.listdir(context)) > 0:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Failed to load project from git, context directory is not empty. "
                    "Set clone param to True to remove the contents of the context directory."
                )
            try:
                repo = Repo(context)
                return get_repo_url(repo), repo
            except Exception:
                pass

    host = url_obj.hostname or "github.com"
    if url_obj.port:
        host += f":{url_obj.port}"

    clone_path = f"https://{host}{url_obj.path}"
    final_clone_path, is_path_enriched = add_credentials_git_remote_url(
        clone_path, secrets=secrets or {}
    )

    branch = None
    tag = None
    commit = None
    if url_obj.fragment:
        refs = url_obj.fragment
        if refs.startswith("refs/heads/"):
            branch = refs.replace("refs/heads/", "")
        elif refs.startswith("refs/tags/"):
            tag = refs.replace("refs/tags/", "")
        elif refs.startswith("refs/commits/"):
            commit = refs.replace("refs/commits/", "")
        else:
            url = url.replace(f"#{refs}", f"#refs/heads/{refs}")
            branch = refs

    # when using the CLI and clone path was not enriched, username/password input will be requested via shell
    repo = Repo.clone_from(final_clone_path, context, single_branch=True, b=branch)

    if is_path_enriched:
        # override enriched clone path for security reasons
        repo.remotes[0].set_url(clone_path, final_clone_path)

    if tag_or_commit := tag or commit:
        repo.git.checkout(tag_or_commit)

    return url, repo


def extract_source(source: str, workdir=None, secrets=None, clone=True, project=None):
    if not source:
        return
    if is_store_uri(source):
        # `clone` is git/archive-specific (it controls whether to wipe the
        # target dir before fetching) — not applicable to store:// URIs,
        # where _load_store_artifact downloads a single file.
        target_dir = workdir or os.path.realpath("./code")
        return load_source_code(
            source_uri=source,
            target_dir=target_dir,
            project=project,
            secrets=secrets,
        )
    clone = clone if workdir else False
    target_dir = workdir or os.path.realpath("./code")
    if source.endswith(".zip"):
        clone_zip(source, target_dir, secrets, clone)
    elif source.endswith(".tar.gz"):
        clone_tgz(source, target_dir, secrets, clone)
    elif source.startswith("git://"):
        clone_git(source, target_dir, secrets, clone)
    else:
        if path.exists(source) and path.isdir(source):
            if workdir and workdir != source:
                raise ValueError("cannot specify both source and workdir")
            return path.realpath(source)
        raise ValueError(f"unsupported source format/path {source}")

    logger.info(f"extracting source from {source} to {target_dir}")
    return target_dir


def load_source_code(
    source_uri: str,
    target_dir: str,
    project: str | None = None,
    secrets=None,
) -> str:
    """
    Load source code from various sources into a target directory.
    This function is used by the Application Runtime init container to prepare
    source code on a shared volume before the sidecar container starts.

    Supported source types:
    - store:// URIs: Single-file artifacts from the MLRun artifact store
    - git:// URLs: Git repositories (cloned to target directory)
    - .zip files: ZIP archives (extracted to target directory)
    - .tar.gz files: Tarball archives (extracted to target directory)

    :param source_uri: Source URI (store://, git://, or archive URL)
    :param target_dir: Target directory where source will be placed
    :param project:    Optional project name (used for store:// URIs)
    :param secrets:    Optional secrets used to access secured data stores
                       (forwarded to the store:// resolver)

    :returns: Path to the directory containing the loaded source.
    """
    if not source_uri:
        raise mlrun.errors.MLRunInvalidArgumentError("source_uri is required")
    if not target_dir:
        raise mlrun.errors.MLRunInvalidArgumentError("target_dir is required")

    # Handle store:// artifact URIs
    if mlrun.datastore.is_store_uri(source_uri):
        return _load_store_artifact(source_uri, target_dir, project, secrets=secrets)

    # Handle git:// URLs
    if source_uri.startswith("git://"):
        return _load_git_source(source_uri, target_dir)

    # Handle archive files (.zip, .tar.gz)
    if source_uri.endswith(".zip") or source_uri.endswith(".tar.gz"):
        return _load_archive_source(source_uri, target_dir)

    raise mlrun.errors.MLRunInvalidArgumentError(
        f"Unsupported source type: {source_uri}. "
        "Supported types: store:// URIs, git:// URLs, .zip and .tar.gz archives"
    )


def _load_store_artifact(
    source_uri: str,
    target_dir: str,
    project: str | None = None,
    secrets=None,
) -> str:
    """
    Load a single-file artifact from the MLRun artifact store.

    :param source_uri: Artifact URI (store://artifacts/project/key)
    :param target_dir: Target directory where the file will be placed
    :param project:    Optional project name (extracted from URI if not provided)
    :param secrets:    Optional secrets used to access secured data stores
                       (forwarded to the artifact resolver and the data-item
                       download for cases where the artifact target path lives
                       on a credential-protected store)

    :returns: Path to the directory containing the loaded source file.
    """
    # Resolve the artifact from the store
    artifact = mlrun.datastore.get_store_resource(
        source_uri,
        project=project,
        secrets=secrets,
        data_store_secrets=secrets,
    )

    # Get the target path where the artifact content is stored
    artifact_target_path = artifact.get_target_path()
    if not artifact_target_path:
        raise ValueError(f"Artifact {source_uri} does not have a valid target path")

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Preserve the original filename uploaded
    if artifact.spec.src_path:
        filename = os.path.basename(artifact.spec.src_path)
    else:
        # src_path may be unset for artifacts created via API/DB directly or with inline body.
        # Fall back to the artifact-store filename
        filename = os.path.basename(artifact_target_path)
    local_file_path = os.path.join(target_dir, filename)

    # Download the artifact content to the target directory
    try:
        mlrun.get_dataitem(artifact_target_path, secrets=secrets).download(
            local_file_path
        )
    except Exception as exc:
        raise mlrun.errors.MLRunRuntimeError(
            f"Failed to download artifact from {artifact_target_path} to {local_file_path}"
        ) from exc

    # Return the directory (not the file path) so that callers like _pre_run()
    # can set it as the working directory and add it to sys.path for imports.
    return target_dir


def _load_git_source(source_uri: str, target_dir: str) -> str:
    """
    Clone a Git repository into the target directory.

    Git credentials are automatically retrieved from environment variables
    by clone_git -> add_credentials_git_remote_url -> get_secret_or_env.

    :param source_uri: Git URL (git://github.com/org/repo.git#branch)
    :param target_dir: Target directory for the cloned repository

    :returns: Path to the cloned repository
    """
    os.makedirs(target_dir, exist_ok=True)

    try:
        clone_git(source_uri, target_dir)
    except Exception as exc:
        raise mlrun.errors.MLRunRuntimeError(
            f"Failed to clone Git repository {source_uri} to {target_dir}"
        ) from exc

    return target_dir


def _load_archive_source(source_uri: str, target_dir: str) -> str:
    """
    Extract an archive (ZIP or tar.gz) into the target directory.

    Storage credentials (S3, V3IO, etc.) are automatically retrieved from
    environment variables by get_dataitem -> store._get_secret_or_env.

    :param source_uri: URL to the archive file
    :param target_dir: Target directory for extraction

    :returns: Path to the extracted directory
    """
    os.makedirs(target_dir, exist_ok=True)

    try:
        if source_uri.endswith(".zip"):
            clone_zip(source_uri, target_dir)
        elif source_uri.endswith(".tar.gz"):
            clone_tgz(source_uri, target_dir)
    except Exception as exc:
        raise mlrun.errors.MLRunRuntimeError(
            f"Failed to extract archive {source_uri} to {target_dir}"
        ) from exc

    return target_dir
