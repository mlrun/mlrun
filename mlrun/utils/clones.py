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


def _prep_dir(source, target_dir, suffix, secrets, clone):
    if not target_dir:
        raise ValueError("please specify a target (context) directory for clone")
    if clone and path.exists(target_dir) and path.isdir(target_dir):
        shutil.rmtree(target_dir)

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    mlrun.get_dataitem(source, secrets).download(temp_file)
    return temp_file


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
    url_obj = urlparse(url)
    secrets = secrets or {}
    if not context:
        raise ValueError("please specify a target (context) directory for clone")

    if path.exists(context) and path.isdir(context):
        if clone:
            shutil.rmtree(context)
        else:
            try:
                repo = Repo(context)
                return get_repo_url(repo), repo
            except Exception:
                pass

    host = url_obj.hostname or "github.com"
    if url_obj.port:
        host += f":{url_obj.port}"

    token = url_obj.username or secrets.get("GITHUB_TOKEN") or secrets.get("git_user")
    password = url_obj.password or secrets.get("git_password") or "x-oauth-basic"
    if token:
        clone_path = f"https://{token}:{password}@{host}{url_obj.path}"
    else:
        clone_path = f"https://{host}{url_obj.path}"

    branch = None
    if url_obj.fragment:
        refs = url_obj.fragment
        if refs.startswith("refs/"):
            branch = refs[refs.rfind("/") + 1 :]
        else:
            url = url.replace("#" + refs, f"#refs/heads/{refs}")

    repo = Repo.clone_from(clone_path, context, single_branch=True, b=branch)
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
