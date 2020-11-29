from http import HTTPStatus
from uuid import uuid4

import deepdiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas


def test_projects_crud(db: Session, client: TestClient) -> None:
    name1 = f"prj-{uuid4().hex}"
    project_1 = mlrun.api.schemas.Project(
        name=name1, owner="owner", description="banana"
    )

    # create
    response = client.post("/api/projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_1, response)

    # read
    response = client.get(f"/api/projects/{name1}")
    _assert_project_response(project_1, response)

    # patch
    project_update = mlrun.api.schemas.ProjectPatch(description="lemon")
    response = client.patch(f"/api/projects/{name1}", json=project_update.dict(exclude_unset=True))
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_1, response, extra_exclude={"description"})
    assert project_update.description == response.json()["description"]

    name2 = f"prj-{uuid4().hex}"
    project_2 = mlrun.api.schemas.Project(
        name=name2, owner="owner", description="banana"
    )

    # store
    response = client.put(f"/api/projects/{name2}", json=project_2.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_2, response)

    # list - names only
    response = client.get("/api/projects", params={"format": mlrun.api.schemas.Format.name_only})
    expected = [name1, name2]
    assert expected == response.json()["projects"]

    # list - full
    response = client.get("/api/projects", params={"format": mlrun.api.schemas.Format.full})
    projects_output = mlrun.api.schemas.ProjectsOutput(**response.json())
    expected = [project_1, project_2]
    for index, project in enumerate(projects_output.projects):
        _assert_project(expected[index], project, extra_exclude={"description"})

    # delete
    response = client.delete(f"/api/projects/{name1}")
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # list
    response = client.get("/api/projects", params={"format": mlrun.api.schemas.Format.name_only})
    expected = [name2]
    assert expected == response.json()["projects"]


def _assert_project_response(expected_project: mlrun.api.schemas.Project, response, extra_exclude: set = None):
    project = mlrun.api.schemas.Project(**response.json())
    _assert_project(expected_project, project, extra_exclude)


def _assert_project(expected_project: mlrun.api.schemas.Project, project: mlrun.api.schemas.Project, extra_exclude: set = None):
    exclude = {"id", "created"}
    if extra_exclude:
        exclude.update(extra_exclude)
    assert (
            deepdiff.DeepDiff(
                expected_project.dict(exclude=exclude),
                project.dict(exclude=exclude),
                ignore_order=True,
            )
            == {}
    )
