from http import HTTPStatus
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.run import new_function


def test_tag(db: Session, client: TestClient) -> None:
    prj = "prj7"
    fn_name = "fn_{}".format
    for i in range(7):
        name = fn_name(i)
        fn = new_function(name=name, project=prj).to_dict()
        tag = uuid4().hex
        resp = client.post(f"/api/func/{prj}/{name}?tag={tag}", json=fn)
        assert resp.status_code == HTTPStatus.OK.value, "status create"
    tag = "t1"
    tagged = {fn_name(i) for i in (1, 3, 4)}
    for name in tagged:
        query = {"functions": {"name": name}}
        resp = client.post(f"/api/{prj}/tag/{tag}", json=query)
        assert resp.status_code == HTTPStatus.OK.value, "status tag"

    resp = client.get(f"/api/{prj}/tag/{tag}")
    assert resp.status_code == HTTPStatus.OK.value, "status get tag"
    objs = resp.json()["objects"]
    assert {obj["name"] for obj in objs} == tagged, "tagged"

    resp = client.delete(f"/api/{prj}/tag/{tag}")
    assert resp.status_code == HTTPStatus.OK.value, "delete"
    resp = client.get(f"/api/{prj}/tags")
    assert resp.status_code == HTTPStatus.OK.value, "list tags"
    assert tag not in resp.json()["tags"], "tag not deleted"
