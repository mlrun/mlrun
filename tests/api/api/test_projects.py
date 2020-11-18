from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_project(db: Session, client: TestClient) -> None:
    name1 = f"prj-{uuid4().hex}"
    prj1 = {
        "name": name1,
        "owner": "u0",
        "description": "banana",
        # 'users': ['u1', 'u2'],
    }
    resp = client.post("/api/project", json=prj1)
    assert resp.status_code == HTTPStatus.OK.value, "add"
    resp = client.get(f"/api/project/{name1}")
    out = {key: val for key, val in resp.json()["project"].items() if val}
    # out['users'].sort()
    for key, value in prj1.items():
        assert out[key] == value

    data = {"description": "lemon", "name": name1}
    resp = client.post(f"/api/project/{name1}", json=data)
    assert resp.status_code == HTTPStatus.OK.value, "update"
    resp = client.get(f"/api/project/{name1}")
    assert name1 == resp.json()["project"]["name"], "name after update"

    name2 = f"prj-{uuid4().hex}"
    prj2 = {
        "name": name2,
        "owner": "u0",
        "description": "banana",
        # 'users': ['u1', 'u3'],
    }
    resp = client.post("/api/project", json=prj2)
    assert resp.status_code == HTTPStatus.OK.value, "add (2)"

    resp = client.get("/api/projects")
    expected = {name1, name2}
    assert expected.issubset(set(resp.json()["projects"])), "list"

    resp = client.get("/api/projects?full=true")
    projects = resp.json()["projects"]
    assert {dict} == set(type(p) for p in projects), "dict"
