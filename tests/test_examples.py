# Copyright 2019 Iguazio
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

from subprocess import run
from sys import executable
from pathlib import Path

import pytest

from conftest import examples_path


example_files = [
    'training.py',
]

db_file = Path('/tmp/mlrun-test-examples.sqlite3')
dsn = f'sqlite:///{db_file}'


@pytest.fixture
def db():
    if db_file.exists():
        db_file.unlink()


@pytest.mark.parametrize('fname', example_files)
def test_example(db, fname):
    path = examples_path / fname
    cmd = [
        executable, '-m', 'mlrun', 'run',
        '--rundb', dsn,
        path,
    ]
    out = run(cmd)
    assert out.returncode == 0, 'bad run'
