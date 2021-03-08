import copy
import typing
import unittest.mock
from http import HTTPStatus
from uuid import uuid4

import deepdiff
import mergedeep
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.errors


def test_create_project_failure_already_exists(db: Session, client: TestClient) -> None:
    name1 = f"prj-{uuid4().hex}"
    project_1 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name1),
    )

    # create
    response = client.post("/api/projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project_1, response)

    # create again
    response = client.post("/api/projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.CONFLICT.value


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
    project_patch = {
        "spec": {
            "description": "lemon",
            "desired_state": mlrun.api.schemas.ProjectState.archived,
        }
    }
    response = client.patch(f"/api/projects/{name1}", json=project_patch)
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(
        project_1, response, extra_exclude={"spec": {"description", "desired_state"}}
    )
    assert (
        project_patch["spec"]["description"] == response.json()["spec"]["description"]
    )
    assert (
        project_patch["spec"]["desired_state"]
        == response.json()["spec"]["desired_state"]
    )
    assert project_patch["spec"]["desired_state"] == response.json()["status"]["state"]

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
    _list_project_names_and_assert(client, [name1, name2])

    # list - names only - filter by label existence
    _list_project_names_and_assert(
        client, [name2], params={"label": list(labels_2.keys())[0]}
    )

    # list - names only - filter by label match
    _list_project_names_and_assert(
        client,
        [name2],
        params={"label": f"{list(labels_2.keys())[0]}={list(labels_2.values())[0]}"},
    )

    # list - full
    response = client.get(
        "/api/projects", params={"format": mlrun.api.schemas.Format.full}
    )
    projects_output = mlrun.api.schemas.ProjectsOutput(**response.json())
    expected = [project_1, project_2]
    for index, project in enumerate(projects_output.projects):
        _assert_project(
            expected[index],
            project,
            extra_exclude={"spec": {"description", "desired_state"}},
        )

    # patch project 1 to have the labels as well
    labels_1 = copy.deepcopy(labels_2)
    labels_1.update({"another-label": "another-label-value"})
    project_patch = {"metadata": {"labels": labels_1}}
    response = client.patch(f"/api/projects/{name1}", json=project_patch)
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(
        project_1,
        response,
        extra_exclude={
            "spec": {"description", "desired_state"},
            "metadata": {"labels"},
        },
    )
    assert (
        deepdiff.DeepDiff(
            response.json()["metadata"]["labels"], labels_1, ignore_order=True,
        )
        == {}
    )

    # list - names only - filter by label existence
    _list_project_names_and_assert(
        client, [name1, name2], params={"label": list(labels_2.keys())[0]}
    )

    # list - names only - filter by label existence
    _list_project_names_and_assert(
        client, [name1], params={"label": list(labels_1.keys())[1]}
    )

    # list - names only - filter by state
    _list_project_names_and_assert(
        client, [name1], params={"state": mlrun.api.schemas.ProjectState.archived}
    )

    # add function to project 1
    function_name = "function-name"
    function = {"metadata": {"name": function_name}}
    response = client.post(f"/api/func/{name1}/{function_name}", json=function)
    assert response.status_code == HTTPStatus.OK.value

    # delete - restrict strategy, will fail because function exists
    response = client.delete(
        f"/api/projects/{name1}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restrict
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

    # mock runtime resources deletion
    mlrun.api.crud.Runtimes().delete_runtimes = unittest.mock.Mock()

    # delete - cascade strategy, will succeed and delete function
    response = client.delete(
        f"/api/projects/{name1}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.cascade
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # ensure function is gone
    response = client.get(f"/api/func/{name1}/{function_name}")
    assert response.status_code == HTTPStatus.NOT_FOUND.value

    # list
    _list_project_names_and_assert(client, [name2])


def _list_project_names_and_assert(
    client: TestClient, expected_names: typing.List[str], params: typing.Dict = None
):
    params = params or {}
    params["format"] = mlrun.api.schemas.Format.name_only
    # list - names only - filter by state
    response = client.get("/api/projects", params=params,)
    assert expected_names == response.json()["projects"]


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
    exclude = {"id": ..., "metadata": {"created"}, "status": {"state"}}
    if extra_exclude:
        mergedeep.merge(exclude, extra_exclude, strategy=mergedeep.Strategy.ADDITIVE)
    assert (
        deepdiff.DeepDiff(
            expected_project.dict(exclude=exclude),
            project.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
