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

import shutil
from http import HTTPStatus
from os import environ
from pathlib import Path
from time import monotonic, sleep
from urllib.request import URLError, urlopen
from sys import platform


here = Path(__file__).absolute().parent
results = here / 'test_results'
is_ci = 'CI' in environ

shutil.rmtree(results, ignore_errors=True, onerror=None)

rundb_path = f'{results}/rundb'
out_path = f'{results}/out'
root_path = str(Path(here).parent)
examples_path = Path(here).parent.joinpath('examples')
environ['PYTHONPATH'] = root_path
environ['MLRUN_DBPATH'] = rundb_path

Path(f'{results}/kfp').mkdir(parents=True, exist_ok=True)
environ['KFPMETA_OUT_DIR'] = f'{results}/kfp/'


def check_docker():
    if not platform.startswith('linux'):
        return False

    with open('/proc/1/cgroup') as fp:
        for line in fp:
            if '/docker/' in line:
                return True
    return False


in_docker = check_docker()

# This must be *after* environment changes above
from mlrun import RunObject, RunTemplate  # noqa


def tag_test(spec: RunTemplate, name) -> RunTemplate:
    spec = spec.copy()
    spec.metadata.name = name
    spec.metadata.labels['test'] = name
    return spec


def has_secrets():
    return Path('secrets.txt').is_file()


def verify_state(result: RunObject):
    state = result.status.state
    assert state == 'completed', \
        'wrong state ({}) {}'.format(state, result.status.error)


def wait_for_server(url, timeout_sec):
    start = monotonic()
    while monotonic() - start <= timeout_sec:
        try:
            with urlopen(url) as resp:
                if resp.status == HTTPStatus.OK:
                    return True
        except (URLError, ConnectionError):
            pass
        sleep(0.1)
    return False
