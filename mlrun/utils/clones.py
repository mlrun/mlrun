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
import os
import shutil
import tarfile
import tempfile
import zipfile
from os import path, remove
from urllib.parse import urlparse

from git import Repo

import mlrun

from .helpers import logger


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
    if token.startswith("github_pat_"):
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


def clone_git(url, context, secrets=None, clone=True):

    secrets = secrets or {}

    def get_secret(key):
        return mlrun.get_secret_or_env(key, secret_provider=secrets)

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

    username = url_obj.username or get_secret("GIT_USERNAME") or get_secret("git_user")
    password = (
        url_obj.password
        or get_secret("GIT_PASSWORD")
        or get_secret("git_password")
        or ""
    )
    token = get_secret("GIT_TOKEN")
    if token:
        username, password = get_git_username_password_from_token(token)

    clone_path = f"https://{host}{url_obj.path}"
    enriched_clone_path = ""
    if username:
        enriched_clone_path = f"https://{username}:{password}@{host}{url_obj.path}"

    branch = None
    tag = None
    if url_obj.fragment:
        refs = url_obj.fragment
        if refs.startswith("refs/heads/"):
            branch = refs.replace("refs/heads/", "")
        elif refs.startswith("refs/tags/"):
            tag = refs.replace("refs/tags/", "")
        else:
            url = url.replace("#" + refs, f"#refs/heads/{refs}")
            branch = refs

    # when using the CLI and clone path was not enriched, username/password input will be requested via shell
    repo = Repo.clone_from(
        enriched_clone_path or clone_path, context, single_branch=True, b=branch
    )

    if enriched_clone_path:

        # override enriched clone path for security reasons
        repo.remotes[0].set_url(clone_path, enriched_clone_path)

    if tag:
        repo.git.checkout(tag)

    return url, repo


def extract_source(source: str, workdir=None, secrets=None, clone=True):
    if not source:
        return
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
