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
