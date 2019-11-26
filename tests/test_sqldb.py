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


def new_func(labels, **kw):
    obj = {
        'metadata': {
            'labels': labels,
        },
    }
    obj.update(kw)
    return obj


def test_list_functions(db: sqldb.SQLDB):
    name = 'fn'
    fn1 = new_func(['l1', 'l2'], x=1)
    db.store_function(fn1, name)
    fn2 = new_func(['l2', 'l3'], x=2)
    db.store_function(fn2, name)
    fn3 = new_func(['l3'], x=3)
    db.store_function(fn3, name)

    funcs = db.list_functions(name, labels=['l2'])
    assert 2 == len(funcs), 'num of funcs'
    assert {1, 2} == {fn['x'] for fn in funcs}, 'xs'


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


def new_run(state, labels, **kw):
    obj = {
        'metadata': {
            'labels': labels,
        },
        'status': {
            'state': state,
        },
    }
    obj.update(kw)
    return obj


def test_runs(db: sqldb.SQLDB):
    run1 = new_run('s1', ['l1', 'l2'], x=1)
    db.store_run(run1, 'uid1')
    run2 = new_run('s1', ['l2', 'l3'], x=2)
    db.store_run(run2, 'uid2')
    run3 = new_run('s2', ['l3'], x=2)
    uid3 = 'uid3'
    db.store_run(run3, uid3)

    runs = db.list_runs(labels=['l2'])
    assert 2 == len(runs), 'labels length'
    assert {1, 2} == {r['x'] for r in runs}, 'xs labels'

    runs = db.list_runs(state='s2')
    assert 1 == len(runs), 'state length'
    assert run3 == runs[0], 'state run'

    db.del_run(uid3)
    with pytest.raises(sqldb.RunDBError):
        db.read_run(uid3)

    label = 'l1'
    db.del_runs(labels=[label])
    for run in db.list_runs():
        assert label not in run['metadata']['labels'], 'del_runs'


def test_artifacts(db: sqldb.SQLDB):
    k1, u1, art1 = 'k1', 'u1', {'a': 1}
    db.store_artifact(k1, art1, u1)
    art = db.read_artifact(k1)
    assert art1['a'] == art['a'], 'get artifact'

    prj = 'p1'
    k2, u2, art2 = 'k2', 'u2', {'a': 2}
    db.store_artifact(k2, art2, u2, project=prj)
    k3, u3, art3 = 'k3', 'u3', {'a': 3}
    db.store_artifact(k3, art3, u3, project=prj)

    arts = db.list_artifacts(project=prj)
    assert 2 == len(arts), 'list artifacts length'
    assert {2, 3} == {a['a'] for a in arts}, 'list artifact a'

    db.del_artifact(key=k1)
    with pytest.raises(sqldb.RunDBError):
        db.read_artifact(k1)
