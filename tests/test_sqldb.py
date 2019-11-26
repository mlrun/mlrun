import pytest

from mlrun.db import sqldb


@pytest.fixture
def db():
    db = sqldb.SQLDB('sqlite:///:memory:')
    db.connect()
    return db


def test_save_get_function(db: sqldb.SQLDB):
    func, name, proj = {'x': 1, 'y': 2}, 'f1', 'p2'
    db.store_function(func, name, proj)
    db_func = db.get_function(name, proj)
    assert func == db_func, 'wrong func'


def test_log(db: sqldb.SQLDB):
    uid = 'm33'
    data1, data2 = b'ab', b'cd'
    db.store_log(uid, body=data1)
    log = db.get_log(uid)
    assert data1 == log, 'get log 1'

    db.store_log(uid, body=data2)
    log = db.get_log(uid)
    assert data1 + data2 == log, 'get log 2'

    db.store_log(uid, body=data1, append=False)
    log = db.get_log(uid)
    assert data1 == log, 'get log append=False'
