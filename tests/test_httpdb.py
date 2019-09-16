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

here = Path(__file__).absolute().parent
Server = namedtuple('Server', 'process port')


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
    env['MLRUN_HTTPDB_PORT'] = port
    env['MLRUN_HTTPDB_DIRPATH'] = dirpath
    fp = open(log_file, 'w')

    cmd = [
        executable,
        f'{here}/../mlrun/db/httpd.py',
    ]
    proc = Popen(cmd, env=env, stdout=fp, stderr=fp)
    url = 'http://localhost:{port}/helathz'
    timeout = 30
    if not wait_for_server(url, timeout):
        raise RuntimeError('server did not start after {timeout}sec')
    return Server(proc, port)
