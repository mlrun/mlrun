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

from contextlib import contextmanager
from os import environ
from subprocess import PIPE, run
from uuid import uuid4

import pytest

from mlrun.db.httpdb import HTTPRunDB
from tests.conftest import in_docker, tests_root_directory, wait_for_server

prj_dir = tests_root_directory.parent
is_ci = "CI" in environ


should_run = (not in_docker) and is_ci


@contextmanager
def clean_docker(cmd, tag):
    yield
    run(["docker", cmd, "-f", tag])


@pytest.mark.skipif(not should_run, reason="in docker container or not CI")
def test_docker():
    tag = f"mlrun/test-{uuid4().hex}"
    cid = None

    cmd = ["docker", "build", "-f", "dockerfiles/mlrun-api/Dockerfile", "-t", tag, "."]
    run(cmd, cwd=prj_dir, check=True)
    with clean_docker("rmi", tag):
        port = 8080
        cmd = ["docker", "run", "--detach", "-p", f"{port}:{port}", tag]
        out = run(cmd, stdout=PIPE, check=True)
        cid = out.stdout.decode("utf-8").strip()
        with clean_docker("rm", cid):
            url = f"http://localhost:{port}/{HTTPRunDB.get_api_path_prefix()}/healthz"
            timeout = 30
            assert wait_for_server(
                url, timeout
            ), f"server failed to start after {timeout} seconds, url={url}"
