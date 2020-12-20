from http import HTTPStatus
from uuid import uuid4

import deepdiff
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas


def test_projects_crud(db: Session, client: TestClient) -> None:
    name1 = f"prj-{uuid4().hex}"
    project_1 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name1),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )

    # create
    response = client.post("/api/projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_1, response)

    # read
    response = client.get(f"/api/projects/{name1}")
    _assert_project_response(project_1, response)

    # patch
    project_patch = {"spec": {"description": "lemon"}}
    response = client.patch(f"/api/projects/{name1}", json=project_patch)
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(
        project_1, response, extra_exclude={"spec": {"description"}}
    )
    assert (
        project_patch["spec"]["description"] == response.json()["spec"]["description"]
    )

    name2 = f"prj-{uuid4().hex}"
    labels_2 = {"key": "value"}
    project_2 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name2, labels=labels_2),
        spec=mlrun.api.schemas.ProjectSpec(description="banana2", source="source2"),
    )

    # store
    response = client.put(f"/api/projects/{name2}", json=project_2.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_2, response)

    # list - names only
    response = client.get(
        "/api/projects", params={"format": mlrun.api.schemas.Format.name_only}
    )
    expected = [name1, name2]
    assert expected == response.json()["projects"]

    # list - names only - filter by label existence
    response = client.get(
        "/api/projects",
        params={
            "format": mlrun.api.schemas.Format.name_only,
            "label": list(labels_2.keys())[0],
        },
    )
    expected = [name2]
    assert expected == response.json()["projects"]

    # list - names only - filter by label match
    response = client.get(
        "/api/projects",
        params={
            "format": mlrun.api.schemas.Format.name_only,
            "label": f"{list(labels_2.keys())[0]}={list(labels_2.values())[0]}",
        },
    )
    expected = [name2]
    assert expected == response.json()["projects"]

    # list - full
    response = client.get(
        "/api/projects", params={"format": mlrun.api.schemas.Format.full}
    )
    projects_output = mlrun.api.schemas.ProjectsOutput(**response.json())
    expected = [project_1, project_2]
    for index, project in enumerate(projects_output.projects):
        _assert_project(
            expected[index], project, extra_exclude={"spec": {"description"}}
        )

    # delete
    response = client.delete(f"/api/projects/{name1}")
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # list
    response = client.get(
        "/api/projects", params={"format": mlrun.api.schemas.Format.name_only}
    )
    expected = [name2]
    assert expected == response.json()["projects"]


def _assert_project_response(
    expected_project: mlrun.api.schemas.Project, response, extra_exclude: dict = None
):
    project = mlrun.api.schemas.Project(**response.json())
    _assert_project(expected_project, project, extra_exclude)


def _assert_project(
    expected_project: mlrun.api.schemas.Project,
    project: mlrun.api.schemas.Project,
    extra_exclude: dict = None,
):
    exclude = {"id": ..., "metadata": {"created"}}
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
