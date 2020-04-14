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
"""SQLDB specific tests, common tests should be in test_dbs.py"""

from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from mlrun.db import sqldb
from conftest import new_run


@pytest.fixture
def db():
    db = sqldb.SQLDB('sqlite:///:memory:?check_same_thread=false')
    db.connect()
    return db


@contextmanager
def patch(obj, **kw):
    old = {}
    for k, v in kw.items():
        old[k] = getattr(obj, k)
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def test_list_artifact_tags(db: sqldb.SQLDB):
    db.store_artifact('k1', {}, '1', tag='t1', project='p1')
    db.store_artifact('k1', {}, '2', tag='t2', project='p1')
    db.store_artifact('k1', {}, '2', tag='t2', project='p2')

    tags = db.list_artifact_tags('p1')
    assert {'t1', 't2'} == set(tags), 'bad tags'


def test_list_artifact_date(db: sqldb.SQLDB):
    t1 = datetime(2020, 2, 16)
    t2 = t1 - timedelta(days=7)
    t3 = t2 - timedelta(days=7)
    prj = 'p7'

    db.store_artifact('k1', {'updated': t1}, 'u1', project=prj)
    db.store_artifact('k2', {'updated': t2}, 'u2', project=prj)
    db.store_artifact('k3', {'updated': t3}, 'u3', project=prj)

    arts = db.list_artifacts(project=prj, since=t3, tag='*')
    assert 3 == len(arts), 'since t3'

    arts = db.list_artifacts(project=prj, since=t2, tag='*')
    assert 2 == len(arts), 'since t2'

    arts = db.list_artifacts(
        project=prj, since=t1 + timedelta(days=1), tag='*')
    assert not arts, 'since t1+'

    arts = db.list_artifacts(project=prj, until=t2, tag='*')
    assert 2 == len(arts), 'until t2'

    arts = db.list_artifacts(project=prj, since=t2, until=t2, tag='*')
    assert 1 == len(arts), 'since/until t2'


def test_list_projects(db: sqldb.SQLDB):
    for i in range(10):
        run = new_run('s1', {'l1': 'v1', 'l2': 'v2'}, x=1)
        db.store_run(run, 'u7', project=f'prj{i%3}', iter=i)

    assert {'prj0', 'prj1', 'prj2'} == {p.name for p in db.list_projects()}


def test_schedules(db: sqldb.SQLDB):
    count = 7
    for i in range(count):
        data = {'i': i}
        db.store_schedule(data)

    scheds = list(db.list_schedules())
    assert count == len(scheds), 'wrong number of schedules'
    assert set(range(count)) == set(s['i'] for s in scheds), 'bad scheds'


def test_run_iter0(db: sqldb.SQLDB):
    uid, prj = 'uid39', 'lemon'
    run = new_run('s1', {'l1': 'v1', 'l2': 'v2'}, x=1)
    for i in range(7):
        db.store_run(run, uid, prj, i)
    db._get_run(uid, prj, 0)  # See issue 140


def test_artifacts_latest(db: sqldb.SQLDB):
    k1, u1, art1 = 'k1', 'u1', {'a': 1}
    prj = 'p38'
    db.store_artifact(k1, art1, u1, project=prj)

    arts = db.list_artifacts(project=prj, tag='latest')
    assert art1['a'] == arts[0]['a'], 'bad artifact'

    u2, art2 = 'u2', {'a': 17}
    db.store_artifact(k1, art2, u2, project=prj)
    arts = db.list_artifacts(project=prj, tag='latest')
    assert 1 == len(arts), 'count'
    assert art2['a'] == arts[0]['a'], 'bad artifact'

    k2, u3, art3 = 'k2', 'u3', {'a': 99}
    db.store_artifact(k2, art3, u3, project=prj)
    arts = db.list_artifacts(project=prj, tag='latest')
    assert 2 == len(arts), 'number'
    assert {17, 99} == set(art['a'] for art in arts), 'latest'


@pytest.mark.parametrize('cls', sqldb._tagged)
def test_tags(db: sqldb.SQLDB, cls):
    p1, n1 = 'prj1', 'name1'
    obj1, obj2, obj3 = cls(), cls(), cls()
    db.session.add(obj1)
    db.session.add(obj2)
    db.session.add(obj3)
    db.session.commit()

    db.tag_objects([obj1, obj2], p1, n1)
    objs = db.find_tagged(p1, n1)
    assert {obj1, obj2} == set(objs), 'find tags'

    db.del_tag(p1, n1)
    objs = db.find_tagged(p1, n1)
    assert [] == objs, 'find tags after del'


def tag_objs(db, count, project, tags):
    by_tag = defaultdict(list)
    for i in range(count):
        cls = sqldb._tagged[i % len(sqldb._tagged)]
        obj = cls()
        by_tag[tags[i % len(tags)]].append(obj)
        db.session.add(obj)
    db.session.commit()
    for tag, objs in by_tag.items():
        db.tag_objects(objs, project, tag)


def test_list_tags(db: sqldb.SQLDB):
    p1, tags1 = 'prj1', ['a', 'b', 'c']
    tag_objs(db, 17, p1, tags1)
    p2, tags2 = 'prj2', ['b', 'c', 'd', 'e']
    tag_objs(db, 11, p2, tags2)

    tags = db.list_tags(p1)
    assert set(tags) == set(tags1), 'tags'


def test_projects(db: sqldb.SQLDB):
    prj1 = {
        'name': 'p1',
        'description': 'banana',
        # 'users': ['u1', 'u2'],
        'spec': {'company': 'ACME'},
        'state': 'active',
        'created': datetime.now(),
    }
    pid1 = db.add_project(prj1)
    p1 = db.get_project(project_id=pid1)
    assert p1, f'project {pid1} not found'
    out = {
        'name': p1.name,
        'description': p1.description,
        # 'users': sorted(u.name for u in p1.users),
        'spec': p1.spec,
        'state': p1.state,
        'created': p1.created,
    }
    assert prj1 == out, 'bad project'

    data = {'description': 'lemon'}
    db.update_project(p1.name, data)
    p1 = db.get_project(project_id=pid1)
    assert data['description'] == p1.description, 'bad update'

    prj2 = {'name': 'p2'}
    db.add_project(prj2)
    prjs = {p.name for p in db.list_projects()}
    assert {prj1['name'], prj2['name']} == prjs, 'list'


def test_cache_projects(db: sqldb.SQLDB):
    assert 0 == len(db._projects), 'empty cache'
    name = 'prj348'
    db.add_project({'name': name})
    assert {name} == db._projects, 'project'

    mock = Mock()
    with patch(db, add_project=mock):
        db._create_project_if_not_exists(name)
    mock.assert_not_called()

    mock = Mock()
    with patch(db, add_project=mock):
        db._create_project_if_not_exists(name + '-new')
    mock.assert_called_once()


# def test_function_latest(db: sqldb.SQLDB):
#     fn1, t1 = {'x': 1}, 'u83'
#     fn2, t2 = {'x': 2}, 'u23'
#     prj, name = 'p388', 'n3023'
#     db.store_function(fn1, name, prj, t1)
#     db.store_function(fn2, name, prj, t2)
#
#     fn = db.get_function(name, prj, 'latest')
#     assert fn2 == fn, 'latest'
