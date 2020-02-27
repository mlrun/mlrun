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


def test_list_projects(client):
    resp = client.get('/api/projects')
    assert resp.status_code == HTTPStatus.OK, 'status'
    assert resp.json['ok'], 'not ok'


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


@pytest.mark.parametrize('typ', sorted(sqldb._type2tag))
def test_tag(client, typ):
    prj = f'prj-{uuid4().hex}'
    name = f'tag-{uuid4().hex}'
    uid1 = f'uid-{uuid4().hex}'
    resp = client.post(f'/api/tag/{prj}/{typ}/{name}/{uid1}')
    assert resp.status_code == HTTPStatus.OK, 'post status'

    resp = client.get(f'/api/tag/{prj}/{typ}/{name}')
    assert resp.status_code == HTTPStatus.OK, 'get status'
    assert uid1 == resp.json['uid'], 'uid'

    resp = client.post(f'/api/tag/{prj}/{typ}/{name}/{uid1}')
    assert resp.status_code == HTTPStatus.BAD_REQUEST, 'post twice'
    uid2 = f'uid-{uuid4().hex}'
    resp = client.put(f'/api/tag/{prj}/{typ}/{name}/{uid2}')
    assert resp.status_code == HTTPStatus.OK, 'update status'
    resp = client.get(f'/api/tag/{prj}/{typ}/{name}')
    assert resp.status_code == HTTPStatus.OK, 'get status'
    assert uid2 == resp.json['uid'], 'uid after update'
