from http import HTTPStatus
from uuid import uuid4

import pytest

from mlrun.db import httpd
from mlrun.run import new_function


@pytest.fixture
def client():
    old_testing = httpd.app.config['TESTING']
    httpd.app.config['TESTING'] = True
    with httpd.app.test_client() as client:
        yield client

    httpd.app.config['TESTING'] = old_testing


def test_project(client):
    name1 = f'prj-{uuid4().hex}'
    prj1 = {
        'name': name1,
        'owner': 'u0',
        'description': 'banana',
        'users': ['u1', 'u2'],
    }
    resp = client.post('/api/project', json=prj1)
    assert resp.status_code == HTTPStatus.OK, 'add'
    resp = client.get(f'/api/project/{name1}')
    out = {key: val for key, val in resp.json['project'].items() if val}
    out['users'].sort()
    assert prj1 == out, 'get'

    data = {'description': 'lemon'}
    resp = client.post(f'/api/project/{name1}', json=data)
    assert resp.status_code == HTTPStatus.OK, 'update'
    resp = client.get(f'/api/project/{name1}')
    assert name1 == resp.json['project']['name'], 'name after update'

    name2 = f'prj-{uuid4().hex}'
    prj2 = {
        'name': name2,
        'owner': 'u0',
        'description': 'banana',
        'users': ['u1', 'u3'],
    }
    resp = client.post('/api/project', json=prj2)
    assert resp.status_code == HTTPStatus.OK, 'add (2)'

    resp = client.get('/api/projects')
    expected = {name1, name2}
    assert expected == expected & set(resp.json['projects']), 'list'

    resp = client.get('/api/projects?full=true')
    projects = resp.json['projects']
    assert {dict} == set(type(p) for p in projects), 'dict'


def test_list_artifact_tags(client):
    project = 'p11'
    resp = client.get(f'/api/projects/{project}/artifact-tags')
    assert resp.status_code == HTTPStatus.OK, 'status'
    assert resp.json['ok'], 'not ok'
    assert resp.json['project'] == project, 'project'


def test_list_schedules(client):
    resp = client.get(f'/api/schedules')
    assert resp.status_code == HTTPStatus.OK, 'status'
    assert resp.json['ok'], 'not ok'
    assert 'schedules' in resp.json, 'no schedules'


def test_tag(client):
    prj = 'prj7'
    fn_name = 'fn_{}'.format
    for i in range(7):
        name = fn_name(i)
        fn = new_function(name=name, project=prj).to_dict()
        resp = client.post(f'/api/func/{prj}/{name}', json=fn)
        assert resp.status_code == HTTPStatus.OK, 'status create'
    tag = 't1'
    tagged = {fn_name(i) for i in (1, 3, 4)}
    for name in tagged:
        query = {'functions': {'name': name}}
        resp = client.post(f'/api/{prj}/tag/{tag}', json=query)
        assert resp.status_code == HTTPStatus.OK, 'status tag'

    resp = client.get(f'/api/{prj}/tag/{tag}')
    assert resp.status_code == HTTPStatus.OK, 'status get tag'
    objs = resp.json['objects']
    assert {obj['name'] for obj in objs} == tagged, 'tagged'

    resp = client.delete(f'/api/{prj}/tag/{tag}')
    assert resp.status_code == HTTPStatus.OK, 'delete'
    resp = client.get(f'/api/{prj}/tags')
    assert resp.status_code == HTTPStatus.OK, 'list tags'
    assert tag not in resp.json['tags'], 'tag not deleted'
