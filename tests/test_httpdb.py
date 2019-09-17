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
from subprocess import Popen
from sys import executable
from tempfile import mkdtemp
from time import monotonic, sleep
from urllib.request import urlopen, URLError
from http import HTTPStatus

import pytest

from mlrun.artifacts import Artifact
from mlrun.db import HTTPRunDB
from mlrun import RunObject

here = Path(__file__).absolute().parent
Server = namedtuple('Server', 'process url log_file conn')


def free_port():
    with socket() as sock:
        sock.bind(('localhost', 0))
        return sock.getsockname()[1]


def wait_for_server(url, timeout_sec):
    start = monotonic()
    while monotonic() - start <= timeout_sec:
        try:
            with urlopen(url) as resp:
                if resp.status == HTTPStatus.OK:
                    return True
        except URLError:
            pass
        sleep(0.1)
    return False


def start_server(dirpath, log_file):
    port = free_port()
    env = environ.copy()
    env['MLRUN_httpdb__port'] = str(port)
    env['MLRUN_httpdb__dirpath'] = dirpath

    cmd = [
        executable,
        f'{here}/../mlrun/db/httpd.py',
    ]
    proc = Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    url = f'http://localhost:{port}'

    health_url = f'{url}/healthz'
    timeout = 30
    if not wait_for_server(health_url, timeout):
        raise RuntimeError('server did not start after {timeout}sec')

    return proc, url, log_file


@pytest.fixture
def server():
    root = mkdtemp(prefix='mlrun-test')
    print(f'root={root!r}')
    dirpath = f'{root}/db'
    with open(f'{root}/httpd.log', 'w+') as log_file:
        proc, url, log_file = start_server(dirpath, log_file)
        conn = HTTPRunDB(url)
        conn.connect()
        yield Server(proc, url, log_file, conn)
        proc.kill()


def test_log(server: Server):
    db = server.conn
    prj, uid, body = 'p19', '3920', b'log data'
    db.store_log(uid, prj, body)

    data = db.get_log(uid, prj)
    assert data == body, 'bad log data'


def test_run(server: Server):
    db = server.conn
    prj, uid = 'p18', '3i920'
    run = RunObject().to_dict()
    run['metadata'].update({
        'algorithm': 'svm',
        'C': 3,
    })
    db.store_run(run, uid, prj, commit=True)

    data = db.read_run(uid, prj)
    assert data == run, 'read_run'

    new_c = 4
    updates = {'metadata.C': new_c}
    db.update_run(updates, uid, prj)
    data = db.read_run(uid, prj)
    assert data['metadata']['C'] == new_c, 'update_run'

    db.del_run(uid, prj)


def test_runs(server):
    db = server.conn

    runs = db.list_runs()
    assert not runs, 'found runs in new db'
    count = 7

    prj = 'p180'
    for i in range(count):
        uid = f'uid_{i}'
        run = RunObject().to_dict()
        db.store_run(run, uid, prj, commit=True)

    runs = db.list_runs(project=prj)
    assert len(runs) == count, 'bad number of runs'

    db.del_runs(project=prj, state='created')
    runs = db.list_runs(project=prj)
    assert not runs, 'found runs in after delete'


def test_artifact(server):
    db = server.conn

    prj, uid, key, body = 'p7', 'u199', 'k800', 'cucumber'
    artifact = Artifact(key, body)

    db.store_artifact(key, artifact, uid, project=prj)
    # TODO: Need a run file
    # db.del_artifact(key, project=prj)


def test_artifacts(server):
    db = server.conn
    prj, uid, key, body = 'p9', 'u19', 'k802', 'tomato'
    artifact = Artifact(key, body)

    db.store_artifact(key, artifact, uid, project=prj)
    artifacts = db.list_artifacts(project=prj)
    assert len(artifacts) == 1, 'bad number of artifacs'

    db.del_artifacts(project=prj)
    artifacts = db.list_artifacts(project=prj)
    assert len(artifacts) == 0, 'bad number of artifacs after del'
