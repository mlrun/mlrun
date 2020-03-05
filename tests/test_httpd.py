from mlrun.db import httpd, sqldb
import pytest
from http import HTTPStatus
from uuid import uuid4


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


@pytest.mark.skip(reason='FIXME')
def test_tag(client, typ):
    ... # TODO: Create some objects, tag subset ...
