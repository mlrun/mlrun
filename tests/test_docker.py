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

from os import environ
from subprocess import PIPE, run
from uuid import uuid4

import pytest

from conftest import here, in_docker, wait_for_server

prj_dir = here.parent
is_ci = 'CI' in environ


should_run = (not in_docker) and is_ci


@pytest.fixture
def build_docker():
    tag = f'mlrun/test-{uuid4().hex}'
    cid = None

    def build(dockerfile):
        nonlocal cid

        cmd = ['docker', 'build', '-f', dockerfile, '-t', tag, '.']
        run(cmd, cwd=prj_dir, check=True)

        port = 8080
        cmd = ['docker', 'run', '--detach', '-p', f'{port}:{port}', tag]
        out = run(cmd, stdout=PIPE, check=True)
        cid = out.stdout.decode('utf-8').strip()

        url = f'http://localhost:{port}/api/healthz'
        timeout = 30
        assert wait_for_server(url, timeout), \
            f'server failed to start after {timeout} seconds, url={url}'
        return cid

    yield build

    if cid is not None:
        run(['docker', 'rm', '-f', cid])
    run(['docker', 'rmi', '-f', tag])


dockerfiles = ['Dockerfile.db', 'Dockerfile.db-gunicorn']


@pytest.mark.skipif(not should_run, reason='in docker container or not CI')
@pytest.mark.parametrize('dockerfile', dockerfiles)
def test_docker(build_docker, dockerfile):
    build_docker(dockerfile)
