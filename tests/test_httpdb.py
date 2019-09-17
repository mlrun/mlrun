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

from mlrun.db import HTTPRunDB
from mlrun import RunObject

here = Path(__file__).absolute().parent
Server = namedtuple('Server', 'process url log_file')


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
    return Server(proc, url, log_file)


@pytest.fixture
def server():
    root = mkdtemp(prefix='mlrun-test')
    print(f'root={root!r}')
    dirpath = f'{root}/db'
    with open(f'{root}/httpd.log', 'w+') as log_file:
        server = start_server(dirpath, log_file)
        yield server
        server.process.kill()


def test_log(server: Server):
    db = HTTPRunDB(server.url)
    db.connect()
    prj, uid, body = 'p19', '3920', b'log data'
    db.store_log(uid, prj, body)

    data = db.get_log(uid, prj)
    assert data == body, 'bad log data'


def test_run(server: Server):
    db = HTTPRunDB(server.url)
    db.connect()
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
