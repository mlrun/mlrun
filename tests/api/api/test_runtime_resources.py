# Copyright 2023 Iguazio
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
import asyncio
import http
import unittest.mock

import deepdiff
import fastapi.testclient
import sqlalchemy.orm

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import server.api.api.endpoints.runtime_resources
import server.api.crud
import server.api.runtime_handlers
import server.api.utils.auth.verifier


def test_list_runtimes_resources_opa_filtering(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        _,
        _,
        _,
        _,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    _mock_list_resources(monkeypatch, grouped_by_project_runtime_resources_output)
    _mock_opa_filter_and_assert_list_response(
        monkeypatch,
        client,
        grouped_by_project_runtime_resources_output,
        [project_1, project_2],
    )

    _mock_opa_filter_and_assert_list_response(
        monkeypatch,
        client,
        grouped_by_project_runtime_resources_output,
        [project_3],
    )

    _mock_opa_filter_and_assert_list_response(
        monkeypatch,
        client,
        grouped_by_project_runtime_resources_output,
        [project_2],
    )


def test_list_runtimes_resources_group_by_job(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    _mock_list_resources(monkeypatch, grouped_by_project_runtime_resources_output)
    _mock_filter_project_resources_by_permissions(monkeypatch)
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.common.schemas.ListRuntimeResourcesGroupByField.job},
    )
    body = response.json()
    expected_body = {
        project_1: {
            project_1_job_name: grouped_by_project_runtime_resources_output[project_1][
                mlrun.runtimes.RuntimeKinds.job
            ].dict()
        },
        project_2: {
            # dask not here cause no dask in group by job
            project_2_job_name: grouped_by_project_runtime_resources_output[project_2][
                mlrun.runtimes.RuntimeKinds.job
            ].dict()
        },
        project_3: {
            project_3_mpijob_name: grouped_by_project_runtime_resources_output[
                project_3
            ][mlrun.runtimes.RuntimeKinds.mpijob].dict()
        },
    }
    assert (
        deepdiff.DeepDiff(
            body,
            expected_body,
            ignore_order=True,
        )
        == {}
    )


def test_list_runtimes_resources_no_group_by(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    _mock_list_resources(monkeypatch, grouped_by_project_runtime_resources_output)
    _mock_filter_project_resources_by_permissions(monkeypatch)
    response = client.get(
        "projects/*/runtime-resources",
    )
    body = response.json()
    expected_body = [
        mlrun.common.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.job,
            resources=mlrun.common.schemas.RuntimeResources(
                crd_resources=[],
                pod_resources=grouped_by_project_runtime_resources_output[project_1][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources
                + grouped_by_project_runtime_resources_output[project_2][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources,
            ),
        ).dict(),
        mlrun.common.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.dask,
            resources=mlrun.common.schemas.RuntimeResources(
                crd_resources=[],
                pod_resources=grouped_by_project_runtime_resources_output[project_2][
                    mlrun.runtimes.RuntimeKinds.dask
                ].pod_resources,
                service_resources=grouped_by_project_runtime_resources_output[
                    project_2
                ][mlrun.runtimes.RuntimeKinds.dask].service_resources,
            ),
        ).dict(),
        mlrun.common.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.mpijob,
            resources=mlrun.common.schemas.RuntimeResources(
                crd_resources=grouped_by_project_runtime_resources_output[project_3][
                    mlrun.runtimes.RuntimeKinds.mpijob
                ].crd_resources,
                pod_resources=[],
            ),
        ).dict(),
    ]
    assert (
        deepdiff.DeepDiff(
            body,
            expected_body,
            ignore_order=True,
        )
        == {}
    )


def test_list_runtime_resources_no_resources(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    _mock_list_resources(monkeypatch, return_value={})
    _mock_filter_project_resources_by_permissions(monkeypatch, return_value=[])
    response = client.get(
        "projects/*/runtime-resources",
    )
    body = response.json()
    assert body == []
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.common.schemas.ListRuntimeResourcesGroupByField.job},
    )
    body = response.json()
    assert body == {}
    response = client.get(
        "projects/*/runtime-resources",
        params={
            "group-by": mlrun.common.schemas.ListRuntimeResourcesGroupByField.project
        },
    )
    body = response.json()
    assert body == {}

    # with kind filter
    response = client.get(
        "projects/*/runtime-resources",
        params={"kind": mlrun.runtimes.RuntimeKinds.job},
    )
    body = response.json()
    assert body == []


def test_list_runtime_resources_filter_by_kind(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()
    filtered_kind = mlrun.runtimes.RuntimeKinds.job

    _mock_filter_project_resources_by_permissions(monkeypatch)
    runtime_handler = server.api.runtime_handlers.get_runtime_handler(filtered_kind)
    with unittest.mock.patch.object(
        runtime_handler,
        "list_resources",
        return_value=_filter_kind_from_grouped_by_project_runtime_resources_output(
            mlrun.runtimes.RuntimeKinds.job,
            grouped_by_project_runtime_resources_output,
        ),
    ):
        response = client.get(
            "projects/*/runtime-resources",
            params={"kind": mlrun.runtimes.RuntimeKinds.job},
        )
        body = response.json()
        expected_runtime_resources = mlrun.common.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.job,
            resources=mlrun.common.schemas.RuntimeResources(
                crd_resources=[],
                pod_resources=grouped_by_project_runtime_resources_output[project_1][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources
                + grouped_by_project_runtime_resources_output[project_2][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources,
            ),
        ).dict()
        expected_body = [expected_runtime_resources]
        assert (
            deepdiff.DeepDiff(
                body,
                expected_body,
                ignore_order=True,
            )
            == {}
        )


def test_delete_runtime_resources_nothing_allowed(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()
    _mock_list_resources(
        monkeypatch, return_value=grouped_by_project_runtime_resources_output
    )
    _mock_filter_project_resources_by_permissions(monkeypatch, return_value=[])
    _assert_forbidden_responses_in_delete_endpoints(client)


def test_delete_runtime_resources_no_resources(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    _mock_list_resources(monkeypatch, return_value={})
    _mock_filter_project_resources_by_permissions(monkeypatch)
    _assert_empty_responses_in_delete_endpoints(client)


def test_delete_runtime_resources_opa_filtering(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    allowed_projects = [project_1, project_2]
    _mock_list_resources(
        monkeypatch, return_value=grouped_by_project_runtime_resources_output
    )
    _mock_filter_project_resources_by_permissions(
        monkeypatch, return_value=allowed_projects
    )
    _mock_runtime_handlers_delete_resources(
        monkeypatch,
        mlrun.runtimes.RuntimeKinds.runtime_with_handlers(),
        allowed_projects,
    )
    response = client.delete(
        "projects/*/runtime-resources",
    )

    # if at least one project isn't allowed, the response should be forbidden
    assert response.status_code == http.HTTPStatus.FORBIDDEN.value


def test_delete_runtime_resources_with_legacy_builder_pod_opa_filtering(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_1_job_name,
        no_project_builder_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_with_legacy_builder_output()

    allowed_projects = []
    _mock_list_resources(
        monkeypatch, return_value=grouped_by_project_runtime_resources_output
    )
    _mock_filter_project_resources_by_permissions(
        monkeypatch, return_value=allowed_projects
    )

    # no projects are allowed, but there is a non project runtime resource (the legacy builder pod)
    # therefore delete resources will be called, but without filter on project in the label selector
    _mock_runtime_handlers_delete_resources(
        monkeypatch,
        mlrun.runtimes.RuntimeKinds.runtime_with_handlers(),
        allowed_projects,
    )
    response = client.delete(
        "projects/*/runtime-resources",
    )

    # if at least one project isn't allowed, the response should be forbidden
    assert response.status_code == http.HTTPStatus.FORBIDDEN.value


def test_delete_runtime_resources_with_kind(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    kind = mlrun.runtimes.RuntimeKinds.job
    grouped_by_project_runtime_resources_output = (
        _filter_kind_from_grouped_by_project_runtime_resources_output(
            kind, grouped_by_project_runtime_resources_output
        )
    )

    allowed_projects = [project_1, project_3]
    _mock_list_resources(
        monkeypatch, return_value=grouped_by_project_runtime_resources_output
    )
    _mock_filter_project_resources_by_permissions(
        monkeypatch, return_value=allowed_projects
    )
    _mock_runtime_handlers_delete_resources(monkeypatch, [kind], allowed_projects)
    response = client.delete(
        "projects/*/runtime-resources",
        params={"kind": kind},
    )
    body = response.json()
    expected_body = _filter_allowed_projects_and_kind_from_grouped_by_project_runtime_resources_output(
        allowed_projects, kind, grouped_by_project_runtime_resources_output
    )
    assert (
        deepdiff.DeepDiff(
            body,
            expected_body,
            ignore_order=True,
        )
        == {}
    )


def test_delete_runtime_resources_with_object_id(
    monkeypatch, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_output()

    kind = mlrun.runtimes.RuntimeKinds.job
    mock_list_runtimes_output = _filter_allowed_projects_and_kind_from_grouped_by_project_runtime_resources_output(
        [project_1], kind, grouped_by_project_runtime_resources_output, structured=True
    )
    object_id = (
        grouped_by_project_runtime_resources_output[project_1][kind]
        .pod_resources[0]
        .name
    )
    _mock_list_resources(monkeypatch, return_value=mock_list_runtimes_output)

    # allow all
    _mock_filter_project_resources_by_permissions(monkeypatch)
    _mock_runtime_handlers_delete_resources(monkeypatch, [kind], [project_1])
    response = client.delete(
        "projects/*/runtime-resources",
        params={"kind": kind, "object-id": object_id},
    )
    body = response.json()
    expected_body = _filter_allowed_projects_and_kind_from_grouped_by_project_runtime_resources_output(
        [project_1], kind, grouped_by_project_runtime_resources_output, structured=False
    )
    assert (
        deepdiff.DeepDiff(
            body,
            expected_body,
            ignore_order=True,
        )
        == {}
    )


def _mock_runtime_handlers_delete_resources(
    monkeypatch,
    kinds: list[str],
    allowed_projects: list[str],
):
    def _assert_delete_resources_label_selector(
        db,
        db_session,
        label_selector: str = None,
        force: bool = False,
        grace_period: int = mlrun.mlconf.runtime_resources_deletion_grace_period,
    ):
        if allowed_projects:
            assert (
                server.api.api.endpoints.runtime_resources._generate_label_selector_for_allowed_projects(
                    allowed_projects
                )
                in label_selector
            )

    for kind in kinds:
        runtime_handler = server.api.runtime_handlers.get_runtime_handler(kind)
        monkeypatch.setattr(
            runtime_handler, "delete_resources", _assert_delete_resources_label_selector
        )


def _assert_empty_responses_in_delete_endpoints(client: fastapi.testclient.TestClient):
    response = client.delete(
        "projects/*/runtime-resources",
    )
    body = response.json()
    assert body == {}


def _assert_forbidden_responses_in_delete_endpoints(
    client: fastapi.testclient.TestClient,
):
    response = client.delete(
        "projects/*/runtime-resources",
    )
    assert response.status_code == http.HTTPStatus.FORBIDDEN.value


def _generate_grouped_by_project_runtime_resources_with_legacy_builder_output():
    no_project = ""
    project_1 = "project-1"
    project_1_job_name = "project-1-job-name"
    no_project_builder_name = "builder-name"
    grouped_by_project_runtime_resources_output = {
        project_1: {
            mlrun.runtimes.RuntimeKinds.job: mlrun.common.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_1_job_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_1,
                            # using name as uid to make assertions easier later
                            mlrun_constants.MLRunInternalLabels.uid: project_1_job_name,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            )
        },
        no_project: {
            mlrun.runtimes.RuntimeKinds.job: mlrun.common.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=no_project_builder_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.mlrun_class: "build",
                            mlrun_constants.MLRunInternalLabels.task_name: "some-task-name",
                        },
                    )
                ],
                crd_resources=[],
            ),
        },
    }
    return (
        project_1,
        project_1_job_name,
        no_project_builder_name,
        grouped_by_project_runtime_resources_output,
    )


def _generate_grouped_by_project_runtime_resources_output():
    project_1 = "project-1"
    project_2 = "project-2"
    project_3 = "project-3"
    project_1_job_name = "project-1-job-name"
    project_2_job_name = "project-2-job-name"
    project_2_dask_name = "project-2-dask-name"
    project_3_mpijob_name = "project-3-mpijob-name"
    grouped_by_project_runtime_resources_output = {
        project_1: {
            mlrun.runtimes.RuntimeKinds.job: mlrun.common.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_1_job_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_1,
                            # using name as uid to make assertions easier later
                            mlrun_constants.MLRunInternalLabels.uid: project_1_job_name,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            )
        },
        project_2: {
            mlrun.runtimes.RuntimeKinds.dask: mlrun.common.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_2_dask_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_2,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.dask,
                            # no uid cause it's dask
                            mlrun_constants.MLRunInternalLabels.function: project_2_dask_name,
                        },
                    )
                ],
                crd_resources=[],
                service_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_2_dask_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_2,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.dask,
                            # no uid cause it's dask
                            mlrun_constants.MLRunInternalLabels.function: project_2_dask_name,
                        },
                    )
                ],
            ),
            mlrun.runtimes.RuntimeKinds.job: mlrun.common.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_2_job_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_2,
                            # using name as uid to make assertions easier later
                            mlrun_constants.MLRunInternalLabels.uid: project_2_job_name,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            ),
        },
        project_3: {
            mlrun.runtimes.RuntimeKinds.mpijob: mlrun.common.schemas.RuntimeResources(
                pod_resources=[],
                crd_resources=[
                    mlrun.common.schemas.RuntimeResource(
                        name=project_3_mpijob_name,
                        labels={
                            mlrun_constants.MLRunInternalLabels.project: project_3,
                            # using name as uid to make assertions easier later
                            mlrun_constants.MLRunInternalLabels.uid: project_3_mpijob_name,
                            mlrun_constants.MLRunInternalLabels.mlrun_class: mlrun.runtimes.RuntimeKinds.mpijob,
                        },
                    )
                ],
            )
        },
    }
    return (
        project_1,
        project_2,
        project_3,
        project_1_job_name,
        project_2_job_name,
        project_2_dask_name,
        project_3_mpijob_name,
        grouped_by_project_runtime_resources_output,
    )


def _mock_opa_filter_and_assert_list_response(
    monkeypatch,
    client: fastapi.testclient.TestClient,
    grouped_by_project_runtime_resources_output: mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    opa_filter_response,
):
    _mock_filter_project_resources_by_permissions(monkeypatch, opa_filter_response)
    response = client.get(
        "projects/*/runtime-resources",
        params={
            "group-by": mlrun.common.schemas.ListRuntimeResourcesGroupByField.project
        },
    )
    body = response.json()
    expected_body = (
        _filter_allowed_projects_from_grouped_by_project_runtime_resources_output(
            opa_filter_response, grouped_by_project_runtime_resources_output
        )
    )
    assert (
        deepdiff.DeepDiff(
            body,
            expected_body,
            ignore_order=True,
        )
        == {}
    )


def _filter_allowed_projects_and_kind_from_grouped_by_project_runtime_resources_output(
    allowed_projects: list[str],
    filter_kind: str,
    grouped_by_project_runtime_resources_output: mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    structured: bool = False,
):
    filtered_output = (
        _filter_allowed_projects_from_grouped_by_project_runtime_resources_output(
            allowed_projects, grouped_by_project_runtime_resources_output, structured
        )
    )
    return _filter_kind_from_grouped_by_project_runtime_resources_output(
        filter_kind, filtered_output
    )


def _filter_kind_from_grouped_by_project_runtime_resources_output(
    filter_kind: str,
    grouped_by_project_runtime_resources_output: mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
):
    filtered_output = {}
    for (
        project,
        kind_runtime_resources_map,
    ) in grouped_by_project_runtime_resources_output.items():
        for kind, runtime_resources in kind_runtime_resources_map.items():
            if kind == filter_kind:
                filtered_output.setdefault(project, {})[kind] = (
                    grouped_by_project_runtime_resources_output[project][kind]
                )
    return filtered_output


def _filter_allowed_projects_from_grouped_by_project_runtime_resources_output(
    allowed_projects: list[str],
    grouped_by_project_runtime_resources_output: mlrun.common.schemas.GroupedByProjectRuntimeResourcesOutput,
    structured: bool = False,
):
    filtered_output = {}
    for project in allowed_projects:
        if project in grouped_by_project_runtime_resources_output:
            filtered_output[project] = {}
            for (
                kind,
                kind_runtime_resources,
            ) in grouped_by_project_runtime_resources_output[project].items():
                filtered_output[project][kind] = (
                    kind_runtime_resources
                    if structured
                    else kind_runtime_resources.dict()
                )
    return filtered_output


def _mock_list_resources(monkeypatch, return_value=None):
    monkeypatch.setattr(
        server.api.crud.RuntimeResources,
        "list_runtime_resources",
        lambda *args, **kwargs: return_value,
    )


def _mock_filter_project_resources_by_permissions(monkeypatch, return_value=None):
    def _async_mock(self, resource_type, resources, *args, **kwargs):
        result = return_value
        if return_value is None:
            result = resources

        future = asyncio.Future()
        future.set_result(result)
        return future

    monkeypatch.setattr(
        server.api.utils.auth.verifier.AuthVerifier,
        "filter_project_resources_by_permissions",
        _async_mock,
    )
