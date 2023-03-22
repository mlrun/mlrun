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
import typing
import unittest.mock

import deepdiff
import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.api.endpoints.runtime_resources
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.singletons.k8s


def test_list_runtimes_resources_opa_filtering(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    _mock_opa_filter_and_assert_list_response(
        client, grouped_by_project_runtime_resources_output, [project_1, project_2]
    )

    _mock_opa_filter_and_assert_list_response(
        client,
        grouped_by_project_runtime_resources_output,
        [project_3],
    )

    _mock_opa_filter_and_assert_list_response(
        client,
        grouped_by_project_runtime_resources_output,
        [project_2],
    )


def test_list_runtimes_resources_group_by_job(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    # allow all
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        side_effect=lambda _, resources, *args, **kwargs: resources
    )
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.job},
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
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    # allow all
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        side_effect=lambda _, resources, *args, **kwargs: resources
    )
    response = client.get(
        "projects/*/runtime-resources",
    )
    body = response.json()
    expected_body = [
        mlrun.api.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.job,
            resources=mlrun.api.schemas.RuntimeResources(
                crd_resources=[],
                pod_resources=grouped_by_project_runtime_resources_output[project_1][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources
                + grouped_by_project_runtime_resources_output[project_2][
                    mlrun.runtimes.RuntimeKinds.job
                ].pod_resources,
            ),
        ).dict(),
        mlrun.api.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.dask,
            resources=mlrun.api.schemas.RuntimeResources(
                crd_resources=[],
                pod_resources=grouped_by_project_runtime_resources_output[project_2][
                    mlrun.runtimes.RuntimeKinds.dask
                ].pod_resources,
                service_resources=grouped_by_project_runtime_resources_output[
                    project_2
                ][mlrun.runtimes.RuntimeKinds.dask].service_resources,
            ),
        ).dict(),
        mlrun.api.schemas.KindRuntimeResources(
            kind=mlrun.runtimes.RuntimeKinds.mpijob,
            resources=mlrun.api.schemas.RuntimeResources(
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
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value={}
    )

    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=[]
    )
    response = client.get(
        "projects/*/runtime-resources",
    )
    body = response.json()
    assert body == []
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.job},
    )
    body = response.json()
    assert body == {}
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.project},
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
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    runtime_handler = mlrun.runtimes.get_runtime_handler(filtered_kind)
    runtime_handler.list_resources = unittest.mock.Mock(
        return_value=_filter_kind_from_grouped_by_project_runtime_resources_output(
            mlrun.runtimes.RuntimeKinds.job,
            grouped_by_project_runtime_resources_output,
        )
    )
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        side_effect=lambda _, resources, *args, **kwargs: resources
    )
    response = client.get(
        "projects/*/runtime-resources",
        params={"kind": mlrun.runtimes.RuntimeKinds.job},
    )
    body = response.json()
    expected_runtime_resources = mlrun.api.schemas.KindRuntimeResources(
        kind=mlrun.runtimes.RuntimeKinds.job,
        resources=mlrun.api.schemas.RuntimeResources(
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
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )

    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=[]
    )
    _assert_forbidden_responses_in_delete_endpoints(client)


def test_delete_runtime_resources_no_resources(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value={}
    )

    # allow all
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        side_effect=lambda _, resources, *args, **kwargs: resources
    )
    _assert_empty_responses_in_delete_endpoints(client)


def test_delete_runtime_resources_opa_filtering(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )

    allowed_projects = [project_1, project_2]
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=allowed_projects
    )
    _mock_runtime_handlers_delete_resources(
        mlrun.runtimes.RuntimeKinds.runtime_with_handlers(), allowed_projects
    )
    response = client.delete(
        "projects/*/runtime-resources",
    )

    # if at least one project isn't allowed, the response should be forbidden
    assert response.status_code == http.HTTPStatus.FORBIDDEN.value


def test_delete_runtime_resources_with_legacy_builder_pod_opa_filtering(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    (
        project_1,
        project_1_job_name,
        no_project_builder_name,
        grouped_by_project_runtime_resources_output,
    ) = _generate_grouped_by_project_runtime_resources_with_legacy_builder_output()

    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )

    allowed_projects = []
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=allowed_projects
    )
    # no projects are allowed, but there is a non project runtime resource (the legacy builder pod)
    # therefore delete resources will be called, but without filter on project in the label selector
    _mock_runtime_handlers_delete_resources(
        mlrun.runtimes.RuntimeKinds.runtime_with_handlers(), allowed_projects
    )
    response = client.delete(
        "projects/*/runtime-resources",
    )

    # if at least one project isn't allowed, the response should be forbidden
    assert response.status_code == http.HTTPStatus.FORBIDDEN.value


def test_delete_runtime_resources_with_kind(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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
    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )

    allowed_projects = [project_1, project_3]
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=allowed_projects
    )
    _mock_runtime_handlers_delete_resources([kind], allowed_projects)
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
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
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
    mlrun.api.crud.RuntimeResources().list_runtime_resources = unittest.mock.Mock(
        return_value=mock_list_runtimes_output
    )

    # allow all
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        side_effect=lambda _, resources, *args, **kwargs: resources
    )
    _mock_runtime_handlers_delete_resources([kind], [project_1])
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
    kinds: typing.List[str],
    allowed_projects: typing.List[str],
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
                mlrun.api.api.endpoints.runtime_resources._generate_label_selector_for_allowed_projects(
                    allowed_projects
                )
                in label_selector
            )

    for kind in kinds:
        runtime_handler = mlrun.runtimes.get_runtime_handler(kind)
        runtime_handler.delete_resources = unittest.mock.Mock(
            side_effect=_assert_delete_resources_label_selector
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
            mlrun.runtimes.RuntimeKinds.job: mlrun.api.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_1_job_name,
                        labels={
                            "mlrun/project": project_1,
                            # using name as uid to make assertions easier later
                            "mlrun/uid": project_1_job_name,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            )
        },
        no_project: {
            mlrun.runtimes.RuntimeKinds.job: mlrun.api.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=no_project_builder_name,
                        labels={
                            "mlrun/class": "build",
                            "mlrun/task-name": "some-task-name",
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
            mlrun.runtimes.RuntimeKinds.job: mlrun.api.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_1_job_name,
                        labels={
                            "mlrun/project": project_1,
                            # using name as uid to make assertions easier later
                            "mlrun/uid": project_1_job_name,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            )
        },
        project_2: {
            mlrun.runtimes.RuntimeKinds.dask: mlrun.api.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_2_dask_name,
                        labels={
                            "mlrun/project": project_2,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.dask,
                            # no uid cause it's dask
                            "mlrun/function": project_2_dask_name,
                        },
                    )
                ],
                crd_resources=[],
                service_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_2_dask_name,
                        labels={
                            "mlrun/project": project_2,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.dask,
                            # no uid cause it's dask
                            "mlrun/function": project_2_dask_name,
                        },
                    )
                ],
            ),
            mlrun.runtimes.RuntimeKinds.job: mlrun.api.schemas.RuntimeResources(
                pod_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_2_job_name,
                        labels={
                            "mlrun/project": project_2,
                            # using name as uid to make assertions easier later
                            "mlrun/uid": project_2_job_name,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.job,
                        },
                    )
                ],
                crd_resources=[],
            ),
        },
        project_3: {
            mlrun.runtimes.RuntimeKinds.mpijob: mlrun.api.schemas.RuntimeResources(
                pod_resources=[],
                crd_resources=[
                    mlrun.api.schemas.RuntimeResource(
                        name=project_3_mpijob_name,
                        labels={
                            "mlrun/project": project_3,
                            # using name as uid to make assertions easier later
                            "mlrun/uid": project_3_mpijob_name,
                            "mlrun/class": mlrun.runtimes.RuntimeKinds.mpijob,
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
    client: fastapi.testclient.TestClient,
    grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
    opa_filter_response,
):
    mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions = unittest.mock.AsyncMock(
        return_value=opa_filter_response
    )
    response = client.get(
        "projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.project},
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
    allowed_projects: typing.List[str],
    filter_kind: str,
    grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
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
    grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
):
    filtered_output = {}
    for (
        project,
        kind_runtime_resources_map,
    ) in grouped_by_project_runtime_resources_output.items():
        for kind, runtime_resources in kind_runtime_resources_map.items():
            if kind == filter_kind:
                filtered_output.setdefault(project, {})[
                    kind
                ] = grouped_by_project_runtime_resources_output[project][kind]
    return filtered_output


def _filter_allowed_projects_from_grouped_by_project_runtime_resources_output(
    allowed_projects: typing.List[str],
    grouped_by_project_runtime_resources_output: mlrun.api.schemas.GroupedByProjectRuntimeResourcesOutput,
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
