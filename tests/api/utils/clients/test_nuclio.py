# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import http

import deepdiff
import pytest
import requests_mock as requests_mock_package

import mlrun.api.schemas
import mlrun.api.utils.clients.nuclio
import mlrun.config
import mlrun.errors


@pytest.fixture()
async def api_url() -> str:
    api_url = "http://nuclio-dashboard-url:8080"
    mlrun.config.config.nuclio_dashboard_url = api_url
    return api_url


@pytest.fixture()
async def nuclio_client(
    api_url: str,
) -> mlrun.api.utils.clients.nuclio.Client:
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
    project_labels = {
        "some-label": "some-label-value",
    }
    project_annotations = {
        "some-annotation": "some-annotation-value",
    }
    response_body = _generate_project_body(
        project_name,
        project_description,
        project_labels,
        project_annotations,
        with_spec=True,
    )
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_client.get_project(None, project_name)
    assert project.metadata.name == project_name
    assert project.spec.description == project_description
    assert (
        deepdiff.DeepDiff(
            project_labels,
            project.metadata.labels,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            project_annotations,
            project.metadata.annotations,
            ignore_order=True,
        )
        == {}
    )

    # now without description, labels and annotations
    response_body = _generate_project_body(project_name, with_spec=True)
    requests_mock.get(f"{api_url}/api/projects/{project_name}", json=response_body)
    project = nuclio_client.get_project(None, project_name)
    assert project.metadata.name == project_name
    assert project.spec.description is None
    assert project.metadata.labels is None
    assert project.metadata.annotations is None


def test_list_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    mock_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3", "labels": {"key": "value"}},
        {
            "name": "project-name-4",
            "annotations": {"annotation-key": "annotation-value"},
        },
        {
            "name": "project-name-5",
            "description": "project-description-4",
            "labels": {"key2": "value2"},
            "annotations": {"annotation-key2": "annotation-value2"},
        },
    ]
    response_body = {
        mock_project["name"]: _generate_project_body(
            mock_project["name"],
            mock_project.get("description"),
            mock_project.get("labels"),
            mock_project.get("annotations"),
            with_spec=True,
        )
        for mock_project in mock_projects
    }
    requests_mock.get(f"{api_url}/api/projects", json=response_body)
    projects = nuclio_client.list_projects(None)
    for index, project in enumerate(projects.projects):
        assert project.metadata.name == mock_projects[index]["name"]
        assert project.spec.description == mock_projects[index].get("description")
        assert (
            deepdiff.DeepDiff(
                mock_projects[index].get("labels"),
                project.metadata.labels,
                ignore_order=True,
            )
            == {}
        )
        assert (
            deepdiff.DeepDiff(
                mock_projects[index].get("annotations"),
                project.metadata.annotations,
                ignore_order=True,
            )
            == {}
        )


def test_create_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_annotations = {
        "some-annotation": "some-annotation-value",
    }

    def verify_creation(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name,
                    project_description,
                    project_labels,
                    project_annotations,
                    with_namespace=False,
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
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                labels=project_labels,
                annotations=project_annotations,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )


def test_store_project_creation(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_annotations = {
        "some-annotation": "some-annotation-value",
    }

    def verify_store_creation(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name,
                    project_description,
                    project_labels,
                    project_annotations,
                    with_namespace=False,
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
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                labels=project_labels,
                annotations=project_annotations,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )


def test_store_project_update(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_annotations = {
        "some-annotation": "some-annotation-value",
    }
    mocked_project_body = _generate_project_body(project_name, with_spec=True)

    def verify_store_update(request, context):
        assert (
            deepdiff.DeepDiff(
                _generate_project_body(
                    project_name,
                    project_description,
                    project_labels,
                    project_annotations,
                    with_namespace=False,
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
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                labels=project_labels,
                annotations=project_annotations,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )


def test_patch_project(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_annotations = {
        "some-annotation": "some-annotation-value",
    }
    mocked_project_body = _generate_project_body(
        project_name,
        labels={"label-key": "label-value"},
        annotations={"annotation-key": "annotation-value"},
        with_spec=True,
    )

    def verify_patch(request, context):
        # verifying the patch kept the old labels, patched the description, and added the new label
        expected_body = mocked_project_body
        expected_body["spec"]["description"] = project_description
        expected_body["metadata"]["labels"].update(project_labels)
        expected_body["metadata"]["annotations"].update(project_annotations)
        assert (
            deepdiff.DeepDiff(
                expected_body,
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    requests_mock.get(
        f"{api_url}/api/projects/{project_name}", json=mocked_project_body
    )
    requests_mock.put(f"{api_url}/api/projects", json=verify_patch)
    nuclio_client.patch_project(
        None,
        project_name,
        {
            "metadata": {"labels": project_labels, "annotations": project_annotations},
            "spec": {"description": project_description},
        },
    )


def test_patch_project_only_labels(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    project_name = "project-name"
    project_labels = {
        "some-label": "some-label-value",
    }
    mocked_project_body = _generate_project_body(
        project_name,
        labels={"label-key": "label-value"},
    )

    def verify_patch(request, context):
        # verifying the patch kept the old labels, patched the description, and added the new label
        expected_body = mocked_project_body
        expected_body["metadata"]["labels"].update(project_labels)
        assert (
            deepdiff.DeepDiff(
                expected_body,
                request.json(),
                ignore_order=True,
            )
            == {}
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    requests_mock.get(
        f"{api_url}/api/projects/{project_name}", json=mocked_project_body
    )
    requests_mock.put(f"{api_url}/api/projects", json=verify_patch)
    nuclio_client.patch_project(
        None,
        project_name,
        {"metadata": {"labels": project_labels}},
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
        assert (
            request.headers["x-nuclio-delete-project-strategy"]
            == mlrun.api.schemas.DeletionStrategy.default().to_nuclio_deletion_strategy()
        )
        context.status_code = http.HTTPStatus.NO_CONTENT.value

    requests_mock.delete(f"{api_url}/api/projects", json=verify_deletion)
    nuclio_client.delete_project(None, project_name)

    # assert ignoring (and not exploding) on not found
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.NOT_FOUND.value
    )
    nuclio_client.delete_project(None, project_name)

    # assert correctly propagating 412 errors (will be returned when project has functions)
    requests_mock.delete(
        f"{api_url}/api/projects", status_code=http.HTTPStatus.PRECONDITION_FAILED.value
    )
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        nuclio_client.delete_project(None, project_name)


def test_get_dashboard_version(
    api_url: str,
    nuclio_client: mlrun.api.utils.clients.nuclio.Client,
    requests_mock: requests_mock_package.Mocker,
):
    label = "x.x.x"
    response_body = {
        "dashboard": {
            "label": label,
            "gitCommit": "commit sha",
            "os": "linux",
            "arch": "amd64",
        }
    }
    requests_mock.get(f"{api_url}/api/versions", json=response_body)
    nuclio_dashboard_version = nuclio_client.get_dashboard_version()
    assert nuclio_dashboard_version == label


def _generate_project_body(
    name=None,
    description=None,
    labels=None,
    annotations=None,
    with_namespace=True,
    with_spec=False,
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
    if annotations:
        body["metadata"]["annotations"] = annotations
    return body
