import http

import deepdiff
import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.clients.nuclio
import mlrun.config


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://nuclio-dashboard-url:8080"
    mlrun.config.config.nuclio_dashboard_url = api_url
    return api_url


@pytest.fixture()
async def nuclio_client(api_url: str,) -> mlrun.api.utils.clients.nuclio.Client:
    client = mlrun.api.utils.clients.nuclio.Client()
    # force running init again so the configured api url will be used
    client.__init__()
    return client


def test_get_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    response_body = _generate_project_body(
        project_name, project_description, with_spec=True
    )
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_client.get_project(None, project_name)
    assert project.name == project_name
    assert project.description == project_description

    # now without description
    response_body = _generate_project_body(project_name, with_spec=True)
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_client.get_project(None, project_name)
    assert project.name == project_name
    assert project.description is None


def test_list_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3"},
        {"name": "project-name-4", "description": "project-description-4"},
    ]
    response_body = {
        mock_project["name"]: _generate_project_body(
            mock_project["name"], mock_project.get("description"), with_spec=True
        )
        for mock_project in mock_projects
    }
    requests_mock.get(f"{api_url}/api/projects", json=response_body)
    projects = nuclio_client.list_projects(None)
    for index, project in enumerate(projects.projects):
        assert project.name == mock_projects[index]["name"]
        assert project.description == mock_projects[index].get("description")


def test_create_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"

    def verify_creation(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name, project_description, with_namespace=False
                ),
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.CREATED.value

    requests_mock.post(f"{api_url}/api/projects", json=verify_creation)
    nuclio_client.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )


def test_store_project_creation(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"

    def verify_store_creation(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name, project_description, with_namespace=False
                ),
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    # mock project not found so store will create
    requests_mock.get(
        f"{api_url}/api/projects/{project_name}",
        status_code=http.HTTPStatus.NOT_FOUND.value,
    )
    requests_mock.post(f"{api_url}/api/projects", json=verify_store_creation)
    nuclio_client.store_project(
        None,
        project_name,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )


def test_store_project_update(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    mocked_project_body = _generate_project_body(
        project_name, labels={"label-key": "label-value"}, with_spec=True
    )

    def verify_store_update(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name, project_description, with_namespace=False
                ),
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    # mock project response so store will update
    requests_mock.get(
        f"{api_url}/api/projects/{project_name}", json=mocked_project_body
    )
    requests_mock.put(f"{api_url}/api/projects", json=verify_store_update)
    nuclio_client.store_project(
        None,
        project_name,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )


def test_patch_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    mocked_project_body = _generate_project_body(
        project_name, labels={"label-key": "label-value"}, with_spec=True
    )

    def verify_patch(request, context):
        # verifying the patch kept the labels and only patched the description
        expected_body = mocked_project_body
        expected_body["spec"]["description"] = project_description
        assert (
            deepdiff.DeepDiff(expected_body, request.json(), ignore_order=True,) == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    requests_mock.get(
        f"{api_url}/api/projects/{project_name}", json=mocked_project_body
    )
    requests_mock.put(f"{api_url}/api/projects", json=verify_patch)
    nuclio_client.patch_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectPatch(description=project_description),
    )


def test_delete_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"

    def verify_deletion(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(project_name, with_namespace=False),
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    requests_mock.delete(f"{api_url}/api/projects", json=verify_deletion)
    nuclio_client.delete_project(None, project_name)


def _generate_project_body(
    name=None, description=None, labels=None, with_namespace=True, with_spec=False
):
    body = {
        "metadata": {"name": name},
    }
    if description:
        body["spec"] = {"description": description}
    elif with_spec:
        body["spec"] = {}
    if with_namespace:
        body["metadata"]["namespace"] = "default-tenant"
    if labels:
        body["metadata"]["labels"] = labels
    return body
