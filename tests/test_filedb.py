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

from tempfile import mkdtemp

import pytest

from mlrun.db import FileRunDB


@pytest.fixture
def db():
    path = mkdtemp(prefix='mlrun-test')
    db = FileRunDB(dirpath=path)
    db.connect()
    return db


def test_save_get_function(db: FileRunDB):
    func, name, proj = {'x': 1, 'y': 2}, 'f1', 'p2'
    db.store_function(func, name, proj)
    db_func = db.get_function(name, proj)
    assert db_func == func, 'wrong func'


def test_list_fuctions(db: FileRunDB):
    proj = 'p4'
    count = 5
    for i in range(count):
        name = f'func{i}'
        func = {'fid': i}
        db.store_function(func, name, proj)
    db.store_function({}, 'f2', 'p7')

    out = db.list_functions('', proj)
    assert len(out) == count, 'bad list'


def test_schedules(db: FileRunDB):
    count = 7
    for i in range(count):
        data = {'i': i}
        db.store_schedule(data)

    scheds = list(db.list_schedules())
    assert count == len(scheds), 'wrong number of schedules'
    assert set(range(count)) == set(s['i'] for s in scheds), 'bad scheds'
