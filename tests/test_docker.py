from subprocess import PIPE, run
from uuid import uuid4

import pytest

from conftest import here, wait_for_server

prj_dir = here.parent


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

        url = f'http://localhost:{port}/healthz'
        timeout = 30
        assert wait_for_server(url, timeout), \
            f'server failed to start after {timeout} seconds, url={url}'
        return cid

    yield build

    if cid is not None:
        run(['docker', 'rm', '-f', cid])
    run(['docker', 'rmi', '-f', tag])


dockerfiles = ['Dockerfile.db', 'Dockerfile.db-gunicorn']


@pytest.mark.parametrize('dockerfile', dockerfiles)
def test_docker(build_docker, dockerfile):
    build_docker(dockerfile)
