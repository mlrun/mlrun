import http

import deepdiff
import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.projects.consumers.nuclio
import mlrun.api.utils.projects.manager
import mlrun.config


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://nuclio-dashboard-url:8080"
    mlrun.config.config.nuclio_dashboard_url = api_url
    return api_url


@pytest.fixture()
async def nuclio_consumer(
    api_url: str,
) -> mlrun.api.utils.projects.consumers.nuclio.Consumer:
    nuclio_consumer = mlrun.api.utils.projects.consumers.nuclio.Consumer()
    return nuclio_consumer


def test_get_project(
    api_url: str,
    nuclio_consumer: mlrun.api.utils.projects.consumers.nuclio.Consumer,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    response_body = _generate_project_body(
        project_name, project_description, with_spec=True
    )
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_consumer.get_project(None, project_name)
    assert project.project.name == project_name
    assert project.project.description == project_description

    # now without description
    response_body = _generate_project_body(project_name, with_spec=True)
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_consumer.get_project(None, project_name)
    assert project.project.name == project_name
    assert project.project.description is None


def test_list_project(
    api_url: str,
    nuclio_consumer: mlrun.api.utils.projects.consumers.nuclio.Consumer,
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
    projects = nuclio_consumer.list_projects(None)
    for index, project in enumerate(projects.projects):
        assert project.name == mock_projects[index]["name"]
        assert project.description == mock_projects[index].get("description")


def test_create_project(
    api_url: str,
    nuclio_consumer: mlrun.api.utils.projects.consumers.nuclio.Consumer,
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
    nuclio_consumer.create_project(
        None,
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )


def test_update_project(
    api_url: str,
    nuclio_consumer: mlrun.api.utils.projects.consumers.nuclio.Consumer,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"

    def verify_update(request, context):
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

    requests_mock.put(f"{api_url}/api/projects", json=verify_update)
    nuclio_consumer.update_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectUpdate(description=project_description),
    )


def test_delete_project(
    api_url: str,
    nuclio_consumer: mlrun.api.utils.projects.consumers.nuclio.Consumer,
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
    nuclio_consumer.delete_project(None, project_name)


def _generate_project_body(
    name=None, description=None, with_namespace=True, with_spec=False
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
    return body
