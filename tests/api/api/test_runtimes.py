import typing
import unittest.mock

import deepdiff
import fastapi.testclient
import sqlalchemy.orm

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.opa
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

    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    _mock_opa_filter_and_assert_list_response(
        client,
        grouped_by_project_runtime_resources_output,
        [
            (project_1, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.dask),
        ],
    )

    _mock_opa_filter_and_assert_list_response(
        client,
        grouped_by_project_runtime_resources_output,
        [(project_3, mlrun.runtimes.RuntimeKinds.mpijob)],
    )

    _mock_opa_filter_and_assert_list_response(
        client,
        grouped_by_project_runtime_resources_output,
        [
            (project_2, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.dask),
        ],
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

    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    # allow all
    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        return_value=[
            (project_1, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.dask),
            (project_3, mlrun.runtimes.RuntimeKinds.mpijob),
        ]
    )
    response = client.get(
        "/api/projects/*/runtime-resources",
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
    assert deepdiff.DeepDiff(body, expected_body, ignore_order=True,) == {}


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

    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )
    # allow all
    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        return_value=[
            (project_1, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.job),
            (project_2, mlrun.runtimes.RuntimeKinds.dask),
            (project_3, mlrun.runtimes.RuntimeKinds.mpijob),
        ]
    )
    response = client.get("/api/projects/*/runtime-resources",)
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
    assert deepdiff.DeepDiff(body, expected_body, ignore_order=True,) == {}


def test_list_runtime_resources_no_resources(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(return_value={})

    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        return_value=[]
    )
    response = client.get("/api/projects/*/runtime-resources",)
    body = response.json()
    assert body == []
    response = client.get(
        "/api/projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.job},
    )
    body = response.json()
    assert body == {}
    response = client.get(
        "/api/projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.project},
    )
    body = response.json()
    assert body == {}


def test_list_kind_runtime_resources(
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

    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(
        return_value=grouped_by_project_runtime_resources_output
    )

    def _mock_opa_filter_resources_by_permissions(
        resource_type: mlrun.api.schemas.AuthorizationResourceTypes,
        resources: typing.List,
        *args,
        **kwargs,
    ):
        for project, kind in resources:
            assert kind == filtered_kind
        # allow all
        return resources

    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        side_effect=_mock_opa_filter_resources_by_permissions
    )
    response = client.get(f"/api/projects/*/runtime-resources/{filtered_kind}",)
    body = response.json()
    expected_body = mlrun.api.schemas.KindRuntimeResources(
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
    assert deepdiff.DeepDiff(body, expected_body, ignore_order=True,) == {}


def test_list_kind_runtime_resources_no_resources(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    filtered_kind = mlrun.runtimes.RuntimeKinds.job

    mlrun.api.crud.Runtimes().list_runtimes = unittest.mock.Mock(return_value={})

    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        return_value=[]
    )
    response = client.get(f"/api/projects/*/runtime-resources/{filtered_kind}",)
    body = response.json()
    expected_body = mlrun.api.schemas.KindRuntimeResources(
        kind=filtered_kind, resources=mlrun.api.schemas.RuntimeResources()
    ).dict()
    assert deepdiff.DeepDiff(body, expected_body, ignore_order=True,) == {}


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
    mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions = unittest.mock.Mock(
        return_value=opa_filter_response
    )
    response = client.get(
        "/api/projects/*/runtime-resources",
        params={"group-by": mlrun.api.schemas.ListRuntimeResourcesGroupByField.project},
    )
    body = response.json()
    expected_body = {}
    for project, kind in opa_filter_response:
        expected_body.setdefault(project, {})[
            kind
        ] = grouped_by_project_runtime_resources_output[project][kind].dict()
    assert deepdiff.DeepDiff(body, expected_body, ignore_order=True,) == {}
