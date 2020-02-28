from mlrun.db import httpd
import pytest
from http import HTTPStatus


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
