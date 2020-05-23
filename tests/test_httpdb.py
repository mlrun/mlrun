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

from collections import namedtuple
from os import environ
from pathlib import Path
from socket import socket
from subprocess import Popen, run, PIPE, DEVNULL
from sys import executable
from tempfile import mkdtemp
from uuid import uuid4
from shutil import rmtree

import pytest

from mlrun.artifacts import Artifact
from mlrun.db import HTTPRunDB, RunDBError
from mlrun import RunObject
from conftest import wait_for_server, in_docker

project_dir_path = Path(__file__).absolute().parent.parent
Server = namedtuple('Server', 'url conn workdir')

docker_tag = 'mlrun/test-api'


def free_port():
    with socket() as sock:
        sock.bind(('localhost', 0))
        return sock.getsockname()[1]


def check_server_up(url):
    health_url = f'{url}/api/healthz'
    timeout = 30
    if not wait_for_server(health_url, timeout):
        raise RuntimeError(f'server did not start after {timeout} sec')


def create_workdir(root_dir='/tmp'):
    return mkdtemp(prefix='mlrun-test-', dir=root_dir)


def start_server(workdir, env_config: dict):
    port = free_port()
    env = environ.copy()
    env['MLRUN_httpdb__port'] = str(port)
    env['MLRUN_httpdb__dsn'] = f'sqlite:///{workdir}/mlrun.sqlite3?check_same_thread=false'
    env['MLRUN_httpdb__logs_path'] = workdir
    env.update(env_config or {})
    cmd = [
        executable,
        '-m', 'mlrun.api.main',
    ]

    proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE, cwd=project_dir_path)
    url = f'http://localhost:{port}'
    check_server_up(url)

    return proc, url


def docker_fixture():
    container_id, workdir = None, None

    def create(env_config=None):
        nonlocal container_id, workdir

        env_config = {} if env_config is None else env_config
        cmd = [
            'docker', 'build',
            '-f', 'dockerfiles/mlrun-api/Dockerfile',
            '--tag', docker_tag,
            '.',
        ]
        run(cmd, check=True, stdout=PIPE, cwd=project_dir_path)
        workdir = create_workdir(root_dir='/tmp')

        cmd = [
            'docker', 'run',
            '--detach',
            '--publish', '8080',

            # For debugging
            '--volume', f'{workdir}:/tmp',
        ]

        env_config.setdefault('MLRUN_httpdb__logs_path', '/tmp')
        for key, value in env_config.items():
            cmd.extend(['--env', f'{key}={value}'])
        cmd.append(docker_tag)
        out = run(cmd, stdout=PIPE, check=True)
        container_id = out.stdout.decode('utf-8').strip()

        # retrieve container bind port + host
        out = run(['docker', 'port', container_id, '8080'], stdout=PIPE, check=True)
        host = out.stdout.decode('utf-8').strip()

        url = f'http://{host}'
        print(f'api url: {url}')
        check_server_up(url)
        conn = HTTPRunDB(url)
        conn.connect()
        return Server(url, conn, workdir)

    def cleanup():
        if container_id:
            run(['docker', 'rm', '--force', container_id], stdout=DEVNULL)
        if workdir:
            rmtree(workdir)

    return create, cleanup


def server_fixture():
    proc = None
    workdir = None

    def create(env=None):
        nonlocal proc, workdir
        workdir = create_workdir()
        proc, url = start_server(workdir, env)
        conn = HTTPRunDB(url)
        conn.connect()
        return Server(url, conn, workdir)

    def cleanup():
        if proc:
            proc.terminate()
        if workdir:
            rmtree(workdir)

    return create, cleanup


servers = [
    'server',
    'docker',
]


@pytest.fixture(scope='function', params=servers)
def create_server(request):
    if request.param == 'server':
        create, cleanup = server_fixture()
    else:
        create, cleanup = docker_fixture()

    try:
        yield create
    finally:
        cleanup()


def test_log(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid, body = 'p19', '3920', b'log data'
    db.store_log(uid, prj, body)

    state, data = db.get_log(uid, prj)
    assert data == body, 'bad log data'


def test_run(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid = 'p18', '3i920'
    run_as_dict = RunObject().to_dict()
    run_as_dict['metadata'].update({
        'algorithm': 'svm',
        'C': 3,
    })
    db.store_run(run_as_dict, uid, prj)

    data = db.read_run(uid, prj)
    assert data == run_as_dict, 'read_run'

    new_c = 4
    updates = {'metadata.C': new_c}
    db.update_run(updates, uid, prj)
    data = db.read_run(uid, prj)
    assert data['metadata']['C'] == new_c, 'update_run'

    db.del_run(uid, prj)


def test_runs(create_server):
    server: Server = create_server()
    db = server.conn

    runs = db.list_runs()
    assert not runs, 'found runs in new db'
    count = 7

    prj = 'p180'
    run_as_dict = RunObject().to_dict()
    for i in range(count):
        uid = f'uid_{i}'
        db.store_run(run_as_dict, uid, prj)

    runs = db.list_runs(project=prj)
    assert len(runs) == count, 'bad number of runs'

    db.del_runs(project=prj, state='created')
    runs = db.list_runs(project=prj)
    assert not runs, 'found runs in after delete'


def test_artifact(create_server):
    server: Server = create_server()
    db = server.conn

    prj, uid, key, body = 'p7', 'u199', 'k800', 'cucumber'
    artifact = Artifact(key, body)

    db.store_artifact(key, artifact, uid, project=prj)
    # TODO: Need a run file
    # db.del_artifact(key, project=prj)


def test_artifacts(create_server):
    server: Server = create_server()
    db = server.conn
    prj, uid, key, body = 'p9', 'u19', 'k802', 'tomato'
    artifact = Artifact(key, body)

    db.store_artifact(key, artifact, uid, project=prj)
    artifacts = db.list_artifacts(project=prj, tag='*')
    assert len(artifacts) == 1, 'bad number of artifacs'

    db.del_artifacts(project=prj, tag='*')
    artifacts = db.list_artifacts(project=prj, tag='*')
    assert len(artifacts) == 0, 'bad number of artifacs after del'


def test_basic_auth(create_server):
    user, passwd = 'bugs', 'bunny'
    env = {
        'MLRUN_httpdb__user': user,
        'MLRUN_httpdb__password': passwd,
    }
    server: Server = create_server(env)

    db: HTTPRunDB = server.conn

    with pytest.raises(RunDBError):
        db.list_runs()

    db.user = user
    db.password = passwd
    db.list_runs()


def test_bearer_auth(create_server):
    token = 'banana'
    env = {'MLRUN_httpdb__token': token}
    server: Server = create_server(env)

    db: HTTPRunDB = server.conn

    with pytest.raises(RunDBError):
        db.list_runs()

    db.token = token
    db.list_runs()


def test_set_get_function(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    func, name, proj = {'x': 1, 'y': 2}, 'f1', 'p2'
    tag = uuid4().hex
    db.store_function(func, name, proj, tag=tag)
    db_func = db.get_function(name, proj, tag=tag)
    del db_func['metadata']
    assert db_func == func, 'wrong func'


def test_list_functions(create_server):
    server: Server = create_server()
    db: HTTPRunDB = server.conn

    proj = 'p4'
    count = 5
    for i in range(count):
        name = f'func{i}'
        func = {'fid': i}
        tag = uuid4().hex
        db.store_function(func, name, proj, tag=tag)
    db.store_function({}, 'f2', 'p7', tag=uuid4().hex)

    out = db.list_functions(project=proj)
    assert len(out) == count, 'bad list'
