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
import collections.abc
import copy
import datetime
import http
import json.decoder
import os
import unittest.mock
from http import HTTPStatus
from uuid import uuid4

import deepdiff
import fastapi.testclient
import kubernetes.client
import mergedeep
import mlrun_pipelines.common.models
import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.errors
import server.api.api.utils
import server.api.crud
import server.api.main
import server.api.utils.auth.verifier
import server.api.utils.background_tasks
import server.api.utils.clients.log_collector
import server.api.utils.singletons.db
import server.api.utils.singletons.k8s
import server.api.utils.singletons.project_member
import server.api.utils.singletons.scheduler
import tests.api.conftest
import tests.api.utils.clients.test_log_collector
from server.api.db.sqldb.models import (
    ArtifactV2,
    Entity,
    Feature,
    FeatureSet,
    FeatureVector,
    Function,
    Project,
    Run,
    Schedule,
    _classes,
)

ORIGINAL_VERSIONED_API_PREFIX = server.api.main.BASE_VERSIONED_API_PREFIX
FUNCTIONS_API = "projects/{project}/functions/{name}"
LIST_FUNCTION_API = "projects/{project}/functions"


@pytest.fixture(params=["leader", "follower"])
def project_member_mode(request, db: Session) -> str:
    if request.param == "follower":
        mlrun.mlconf.httpdb.projects.leader = "nop"
        server.api.utils.singletons.project_member.initialize_project_member()
        server.api.utils.singletons.project_member.get_project_member()._leader_client.db_session = db
    elif request.param == "leader":
        mlrun.mlconf.httpdb.projects.leader = "mlrun"
        server.api.utils.singletons.project_member.initialize_project_member()
    else:
        raise NotImplementedError(
            f"Provided project member mode is not supported. mode={request.param}"
        )
    yield request.param


def test_redirection_from_worker_to_chief_delete_project(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project_name = "test-project"
    _create_project(client, project_name)

    endpoint = f"projects/{project_name}"
    for strategy in mlrun.common.schemas.DeletionStrategy:
        headers = {"x-mlrun-deletion-strategy": strategy.value}
        for test_case in [
            # deleting schedule failed for unknown reason
            {
                "expected_status": http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "expected_body": {"detail": {"reason": "Unknown error"}},
            },
            # deleting project accepted and is running in background (in follower mode, forwarding request to leader)
            {
                "expected_status": http.HTTPStatus.ACCEPTED.value,
                "expected_body": {},
            },
            # received request from leader and succeeded deleting
            {
                "expected_status": http.HTTPStatus.NO_CONTENT.value,
                "expected_body": "",
            },
            {
                "expected_status": http.HTTPStatus.PRECONDITION_FAILED.value,
                "expected_body": {
                    "detail": {
                        "reason": f"Project {project_name} can not be deleted since related resources found: x"
                    }
                },
            },
        ]:
            expected_status = test_case.get("expected_status")
            expected_response = test_case.get("expected_body")

            httpserver.expect_ordered_request(
                f"{ORIGINAL_VERSIONED_API_PREFIX}/{endpoint}", method="DELETE"
            ).respond_with_json(expected_response, status=expected_status)
            url = httpserver.url_for("")
            mlrun.mlconf.httpdb.clusterization.chief.url = url
            response = client.delete(endpoint, headers=headers)
            assert response.status_code == expected_status
            try:
                assert response.json() == expected_response
            except json.decoder.JSONDecodeError:
                # NO_CONTENT response doesn't return json serializable response
                assert response.text == expected_response


def test_create_project_failure_already_exists(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    name = f"prj-{uuid4().hex}"
    project = _create_project(client, name)

    # create again
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CONFLICT.value


def test_get_non_existing_project(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    """
    At first we were doing auth before get - which caused get on non existing project to return unauthorized instead of
    not found - which "ruined" the `mlrun.get_or_create_project` logic - so adding a specific test to verify it works
    """
    project = "does-not-exist"
    server.api.utils.auth.verifier.AuthVerifier().query_project_permissions = (
        unittest.mock.AsyncMock(side_effect=mlrun.errors.MLRunUnauthorizedError("bla"))
    )
    response = client.get(f"projects/{project}")
    assert response.status_code == HTTPStatus.NOT_FOUND.value


@pytest.fixture()
def mock_process_model_monitoring_secret() -> collections.abc.Iterator[None]:
    with unittest.mock.patch(
        "server.api.api.endpoints.nuclio.process_model_monitoring_secret",
        return_value="some_access_key",
    ):
        yield


@pytest.mark.usefixtures("mock_process_model_monitoring_secret")
@pytest.mark.parametrize(
    "api_version,successful_delete_response_code",
    [("v1", HTTPStatus.NO_CONTENT.value), ("v2", HTTPStatus.ACCEPTED.value)],
)
def test_delete_project_with_resources(
    db: Session,
    unversioned_client: TestClient,
    mocked_k8s_helper,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    project_member_mode: str,
    api_version: str,
    successful_delete_response_code: int,
):
    def _send_delete_request_and_assert_response_code(
        deletion_strategy: mlrun.common.schemas.DeletionStrategy,
        expected_response_code: int,
    ):
        response = unversioned_client.delete(
            f"{api_version}/projects/{project_to_remove}",
            headers={
                mlrun.common.schemas.HeaderNames.deletion_strategy: deletion_strategy.value
            },
        )
        assert response.status_code == expected_response_code

    # need to set this to False, otherwise impl will try to delete k8s resources, and will need many more
    # mocks to overcome this.
    k8s_secrets_mock.set_is_running_in_k8s_cluster(False)
    mlrun.mlconf.namespace = "test-namespace"
    project_to_keep = "project-to-keep"
    project_to_remove = "project-to-remove"
    _create_resources_of_all_kinds(db, k8s_secrets_mock, project_to_keep)
    _create_resources_of_all_kinds(db, k8s_secrets_mock, project_to_remove)

    (
        project_to_keep_table_name_records_count_map_before_project_removal,
        project_to_keep_object_records_count_map_before_project_removal,
    ) = _assert_resources_in_project(
        db, k8s_secrets_mock, project_member_mode, project_to_keep
    )
    _assert_resources_in_project(
        db, k8s_secrets_mock, project_member_mode, project_to_remove
    )

    # deletion strategy - check - should fail because there are resources
    _send_delete_request_and_assert_response_code(
        mlrun.common.schemas.DeletionStrategy.check,
        HTTPStatus.PRECONDITION_FAILED.value,
    )

    # deletion strategy - restricted - should fail because there are resources
    _send_delete_request_and_assert_response_code(
        mlrun.common.schemas.DeletionStrategy.restricted,
        HTTPStatus.PRECONDITION_FAILED.value,
    )

    # deletion strategy - cascading - should succeed and remove all related resources
    # mock project configmaps
    k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()

    def _list_configmaps(*args, **kwargs):
        label_selector = kwargs.get("label_selector")
        assert project_to_remove in label_selector
        return kubernetes.client.V1ConfigMapList(
            items=[
                kubernetes.client.V1ConfigMap(
                    metadata=kubernetes.client.V1ObjectMeta(
                        name=f"{project_to_remove}-configmap",
                    )
                )
            ]
        )

    k8s_helper.v1api.list_namespaced_config_map = unittest.mock.Mock(
        side_effect=_list_configmaps
    )
    k8s_helper.delete_configmap = unittest.mock.Mock()
    _send_delete_request_and_assert_response_code(
        mlrun.common.schemas.DeletionStrategy.cascading,
        successful_delete_response_code,
    )
    k8s_helper.delete_configmap.assert_called_once()

    (
        project_to_keep_table_name_records_count_map_after_project_removal,
        project_to_keep_object_records_count_map_after_project_removal,
    ) = _assert_resources_in_project(
        db, k8s_secrets_mock, project_member_mode, project_to_keep
    )
    _assert_resources_in_project(
        db,
        k8s_secrets_mock,
        project_member_mode,
        project_to_remove,
        assert_no_resources=True,
    )
    assert (
        deepdiff.DeepDiff(
            project_to_keep_object_records_count_map_before_project_removal,
            project_to_keep_object_records_count_map_after_project_removal,
            ignore_order=True,
        )
        == {}
    )
    assert (
        deepdiff.DeepDiff(
            project_to_keep_table_name_records_count_map_before_project_removal,
            project_to_keep_table_name_records_count_map_after_project_removal,
            ignore_order=True,
        )
        == {}
    )

    # deletion strategy - check - should succeed cause no project
    _send_delete_request_and_assert_response_code(
        mlrun.common.schemas.DeletionStrategy.check,
        HTTPStatus.NO_CONTENT.value,
    )

    # deletion strategy - restricted - should succeed cause no project
    _send_delete_request_and_assert_response_code(
        mlrun.common.schemas.DeletionStrategy.restricted,
        HTTPStatus.NO_CONTENT.value,
    )


@pytest.mark.asyncio
async def test_list_and_get_project_summaries(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    # Create projects
    empty_project_name = "empty-project"
    _create_project(client, empty_project_name)
    _create_project(client, "project-with-resources")

    # Create resources for the second project
    project_name = "project-with-resources"

    # create files for the project
    files_count = 5
    _create_artifacts(
        client, project_name, files_count, mlrun.artifacts.PlotArtifact.kind
    )

    # create feature sets for the project
    feature_sets_count = 9
    _create_feature_sets(client, project_name, feature_sets_count)

    # create model artifacts for the project
    models_count = 4
    _create_artifacts(
        client, project_name, models_count, mlrun.artifacts.model.ModelArtifact.kind
    )

    # create dataset artifacts for the project to make sure we're not mistakenly counting them
    _create_artifacts(
        client, project_name, 7, mlrun.artifacts.dataset.DatasetArtifact.kind
    )

    # create runs for the project
    running_runs_count = 5
    _create_runs(
        client,
        project_name,
        running_runs_count,
        mlrun.common.runtimes.constants.RunStates.running,
    )

    # create completed runs for the project to make sure we're not mistakenly counting them
    two_days_ago = datetime.datetime.now() - datetime.timedelta(hours=48)
    _create_runs(
        client,
        project_name,
        2,
        mlrun.common.runtimes.constants.RunStates.completed,
        two_days_ago,
    )

    # create completed runs for the project for less than 24 hours ago
    runs_completed_recent_count = 10
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    _create_runs(
        client,
        project_name,
        runs_completed_recent_count,
        mlrun.common.runtimes.constants.RunStates.completed,
        one_hour_ago,
    )

    # create failed runs for the project for less than 24 hours ago
    recent_failed_runs_count = 6
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    _create_runs(
        client,
        project_name,
        recent_failed_runs_count,
        mlrun.common.runtimes.constants.RunStates.error,
        one_hour_ago,
    )

    # create aborted runs for the project for less than 24 hours ago - make sure we count them as well
    recent_aborted_runs_count = 6
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    _create_runs(
        client,
        project_name,
        recent_failed_runs_count,
        mlrun.common.runtimes.constants.RunStates.aborted,
        one_hour_ago,
    )

    # create failed runs for the project for more than 24 hours ago to make sure we're not mistakenly counting them
    two_days_ago = datetime.datetime.now() - datetime.timedelta(hours=48)
    _create_runs(
        client,
        project_name,
        3,
        mlrun.common.runtimes.constants.RunStates.error,
        two_days_ago,
    )

    # create schedules for the project

    (
        schedules_count,
        distinct_scheduled_jobs_pending_count,
        distinct_scheduled_pipelines_pending_count,
    ) = _create_schedules(
        client,
        project_name,
    )

    # mock pipelines for the project
    running_pipelines_count = _mock_pipelines(
        project_name,
    )

    await server.api.crud.Projects().refresh_project_resources_counters_cache(db)

    # list project summaries
    response = client.get("project-summaries")
    project_summaries_output = mlrun.common.schemas.ProjectSummariesOutput(
        **response.json()
    )
    for index, project_summary in enumerate(project_summaries_output.project_summaries):
        if project_summary.name == empty_project_name:
            _assert_project_summary(project_summary, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        elif project_summary.name == project_name:
            _assert_project_summary(
                project_summary,
                files_count,
                feature_sets_count,
                models_count,
                runs_completed_recent_count,
                recent_failed_runs_count + recent_aborted_runs_count,
                running_runs_count,
                schedules_count,
                distinct_scheduled_jobs_pending_count,
                distinct_scheduled_pipelines_pending_count,
                running_pipelines_count,
            )
        else:
            pytest.fail(f"Unexpected project summary returned: {project_summary}")

    # get project summary
    response = client.get(f"project-summaries/{project_name}")
    project_summary = mlrun.common.schemas.ProjectSummary(**response.json())
    _assert_project_summary(
        project_summary,
        files_count,
        feature_sets_count,
        models_count,
        runs_completed_recent_count,
        recent_failed_runs_count + recent_aborted_runs_count,
        running_runs_count,
        schedules_count,
        distinct_scheduled_jobs_pending_count,
        distinct_scheduled_pipelines_pending_count,
        running_pipelines_count,
    )


@pytest.mark.asyncio
async def test_list_project_summaries_different_installation_modes(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    """
    The list project summaries endpoint is used in our projects screen and tend to break in different installation modes
    """
    # create empty project
    empty_project_name = "empty-project"
    _create_project(client, empty_project_name)

    server.api.crud.Pipelines().list_pipelines = unittest.mock.Mock(
        return_value=(0, None, [])
    )
    # Enterprise installation configuration post 3.4.0
    mlrun.mlconf.igz_version = "3.6.0-b26.20210904121245"
    mlrun.mlconf.kfp_url = "https://somekfp-url.com"
    mlrun.mlconf.namespace = "default-tenant"

    await server.api.crud.Projects().refresh_project_resources_counters_cache(db)

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.common.schemas.ProjectSummariesOutput(
        **response.json()
    )
    _assert_project_summary(
        # accessing the zero index as there's only one project
        project_summaries_output.project_summaries[0],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    # Enterprise installation configuration pre 3.4.0
    mlrun.mlconf.igz_version = "3.2.0-b26.20210904121245"
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = "default-tenant"

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.common.schemas.ProjectSummariesOutput(
        **response.json()
    )
    _assert_project_summary(
        # accessing the zero index as there's only one project
        project_summaries_output.project_summaries[0],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    # Kubernetes installation configuration (mlrun-kit)
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = "mlrun"

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.common.schemas.ProjectSummariesOutput(
        **response.json()
    )
    _assert_project_summary(
        # accessing the zero index as there's only one project
        project_summaries_output.project_summaries[0],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    # Docker installation configuration
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = ""

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.common.schemas.ProjectSummariesOutput(
        **response.json()
    )
    _assert_project_summary(
        # accessing the zero index as there's only one project
        project_summaries_output.project_summaries[0],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_delete_project_deletion_strategy_check(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    project = _create_project(client, "project-name")

    # deletion strategy - check - should succeed because there are no resources
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.check.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # ensure project not deleted
    response = client.get(f"projects/{project.metadata.name}")
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project, response)

    # add function to project 1
    function_name = "function-name"
    function = {"metadata": {"name": function_name}}
    response = client.post(
        FUNCTIONS_API.format(project=project.metadata.name, name=function_name),
        json=function,
    )
    assert response.status_code == HTTPStatus.OK.value

    # deletion strategy - check - should fail because there are resources
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.check.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value


def test_delete_project_not_deleting_versioned_objects_multiple_times(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # need to set this to False, otherwise impl will try to delete k8s resources, and will need many more
    # mocks to overcome this.
    k8s_secrets_mock.set_is_running_in_k8s_cluster(False)
    project_name = "project-name"
    _create_resources_of_all_kinds(db, k8s_secrets_mock, project_name)

    response = client.get(LIST_FUNCTION_API.format(project=project_name))
    assert response.status_code == HTTPStatus.OK.value
    distinct_function_names = {
        function["metadata"]["name"] for function in response.json()["funcs"]
    }
    # ensure there are indeed several versions of the same function name
    assert len(distinct_function_names) < len(response.json()["funcs"])

    response = client.get(f"projects/{project_name}/artifacts", params={"tag": "*"})
    assert response.status_code == HTTPStatus.OK.value
    # ensure there are indeed several versions of the same artifact key
    distinct_artifact_keys = {
        (artifact["spec"]["db_key"], artifact["metadata"]["iter"])
        for artifact in response.json()["artifacts"]
    }
    assert len(distinct_artifact_keys) < len(response.json()["artifacts"])

    response = client.get(
        f"projects/{project_name}/feature-sets",
    )
    assert response.status_code == HTTPStatus.OK.value
    distinct_feature_set_names = {
        feature_set["metadata"]["name"]
        for feature_set in response.json()["feature_sets"]
    }
    # ensure there are indeed several versions of the same feature_set name
    assert len(distinct_feature_set_names) < len(response.json()["feature_sets"])

    response = client.get(
        f"projects/{project_name}/feature-vectors",
    )
    assert response.status_code == HTTPStatus.OK.value
    distinct_feature_vector_names = {
        feature_vector["metadata"]["name"]
        for feature_vector in response.json()["feature_vectors"]
    }
    # ensure there are indeed several versions of the same feature_vector name
    assert len(distinct_feature_vector_names) < len(response.json()["feature_vectors"])

    server.api.utils.singletons.db.get_db().delete_functions = unittest.mock.Mock()
    server.api.utils.singletons.db.get_db().delete_feature_set = unittest.mock.Mock()
    server.api.utils.singletons.db.get_db().delete_feature_vector = unittest.mock.Mock()
    # deletion strategy - check - should fail because there are resources
    response = client.delete(
        f"projects/{project_name}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.cascading.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    assert len(
        server.api.utils.singletons.db.get_db().delete_functions.call_args.args[2]
    ) == len(distinct_function_names)
    assert server.api.utils.singletons.db.get_db().delete_feature_set.call_count == len(
        distinct_feature_set_names
    )
    assert (
        server.api.utils.singletons.db.get_db().delete_feature_vector.call_count
        == len(distinct_feature_vector_names)
    )


def test_delete_project_deletion_strategy_check_external_resource(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    mocked_k8s_helper,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    mlrun.mlconf.namespace = "test-namespace"
    project = _create_project(client, "project-name")

    # Set a project secret
    k8s_secrets_mock.store_project_secrets("project-name", {"secret": "value"})

    # deletion strategy - check - should fail because there's a project secret
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value
    assert "project secrets" in response.text

    k8s_secrets_mock.delete_project_secrets("project-name", None)
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response


def test_delete_project_with_stop_logs(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    mocked_k8s_helper,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
    mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar

    project_name = "project-name"

    mlrun.mlconf.namespace = "test-namespace"
    _create_project(client, project_name)

    log_collector = server.api.utils.clients.log_collector.LogCollectorClient()
    with unittest.mock.patch.object(
        server.api.utils.clients.log_collector.LogCollectorClient,
        "_call",
        return_value=tests.api.utils.clients.test_log_collector.BaseLogCollectorResponse(
            True, ""
        ),
    ):
        # deletion strategy - cascading - should succeed and remove all related resources
        response = client.delete(
            f"projects/{project_name}",
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value

        # 2 calls - stop logs and delete logs
        assert log_collector._call.call_count == 2
        assert log_collector._call.call_args[0][0] == "DeleteLogs"


def test_project_with_invalid_node_selector(
    db: Session,
    client: TestClient,
):
    project_name = "project-name"
    project = _create_project(client, project_name)
    invalid_node_selector = {"invalid": "node=selector"}

    project.spec.default_function_node_selector = invalid_node_selector
    response = client.put(f"projects/{project_name}", json=project.dict())
    assert response.status_code == HTTPStatus.BAD_REQUEST.value

    valid_node_selector = {"label": "val"}
    project.spec.default_function_node_selector = valid_node_selector
    response = client.put(f"projects/{project_name}", json=project.dict())
    assert response.status_code == HTTPStatus.OK.value
    _assert_project_response(project, response)


# leader format is only relevant to follower mode
@pytest.mark.parametrize("project_member_mode", ["follower"], indirect=True)
def test_list_projects_leader_format(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    """
    See list_projects in follower.py for explanation on the rationality behind the leader format
    """
    # create some projects in the db (mocking projects left there from before when leader format was used)
    project_names = []
    for _ in range(5):
        project_name = f"prj-{uuid4().hex}"
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        )
        server.api.utils.singletons.db.get_db().create_project(db, project)
        project_names.append(project_name)

    # list in leader format
    response = client.get(
        "projects",
        params={"format": mlrun.common.formatters.ProjectFormat.leader},
        headers={
            mlrun.common.schemas.HeaderNames.projects_role: mlrun.mlconf.httpdb.projects.leader
        },
    )
    returned_project_names = [
        project["data"]["metadata"]["name"] for project in response.json()["projects"]
    ]
    assert (
        deepdiff.DeepDiff(
            project_names,
            returned_project_names,
            ignore_order=True,
        )
        == {}
    )


def test_projects_crud(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # need to set this to False, otherwise impl will try to delete k8s resources, and will need many more
    # mocks to overcome this.
    k8s_secrets_mock.set_is_running_in_k8s_cluster(False)

    name1 = f"prj-{uuid4().hex}"
    project_1 = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name1),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )

    # create - fail invalid label
    invalid_project_create_request = project_1.dict()
    invalid_project_create_request["metadata"]["labels"] = {".a": "invalid-label"}
    response = client.post("projects", json=invalid_project_create_request)
    assert response.status_code == HTTPStatus.BAD_REQUEST.value

    # create
    response = client.post("projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project_1, response)

    # read
    response = client.get(f"projects/{name1}")
    _assert_project_response(project_1, response)

    # patch
    project_patch = {
        "spec": {
            "description": "lemon",
            "desired_state": mlrun.common.schemas.ProjectState.archived,
        }
    }
    response = client.patch(f"projects/{name1}", json=project_patch)
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
    project_2 = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name2, labels=labels_2),
        spec=mlrun.common.schemas.ProjectSpec(description="banana2", source="source2"),
    )

    # store
    response = client.put(f"projects/{name2}", json=project_2.dict())
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
        "projects", params={"format": mlrun.common.formatters.ProjectFormat.full}
    )
    projects_output = mlrun.common.schemas.ProjectsOutput(**response.json())
    expected = [project_1, project_2]
    for project in projects_output.projects:
        for _project in expected:
            if _project.metadata.name == project.metadata.name:
                _assert_project(
                    _project,
                    project,
                    extra_exclude={"spec": {"description", "desired_state"}},
                )
            expected.remove(_project)
            break

    # patch project 1 to have the labels as well
    labels_1 = copy.deepcopy(labels_2)
    labels_1.update({"another-label": "another-label-value"})
    project_patch = {"metadata": {"labels": labels_1}}
    response = client.patch(f"projects/{name1}", json=project_patch)
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
            response.json()["metadata"]["labels"],
            labels_1,
            ignore_order=True,
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
        client, [name1], params={"state": mlrun.common.schemas.ProjectState.archived}
    )

    # add function to project 1
    function_name = "function-name"
    function = {"metadata": {"name": function_name}}
    response = client.post(
        FUNCTIONS_API.format(project=name1, name=function_name), json=function
    )
    assert response.status_code == HTTPStatus.OK.value

    # delete - restricted strategy, will fail because function exists
    response = client.delete(
        f"projects/{name1}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

    # delete - cascading strategy, will succeed and delete function
    response = client.delete(
        f"projects/{name1}",
        headers={
            mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.cascading.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # ensure function is gone
    response = client.get(FUNCTIONS_API.format(project=name1, name=function_name))
    assert response.status_code == HTTPStatus.NOT_FOUND.value

    # list
    _list_project_names_and_assert(client, [name2])


def test_project_with_parameters(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # validate that leading/trailing whitespaces in the keys and values are removed

    project = _create_project(client, "project-name")

    project.spec.params = {"aa": "1", "aa ": "1", "aa   ": "1", " bb ": "   2"}
    expected_params = {"aa": "1", "bb": "2"}

    # store project request to save the parameters
    response = client.put(f"projects/{project.metadata.name}", json=project.dict())
    assert response.status_code == HTTPStatus.OK.value

    # get project request
    response = client.get(f"projects/{project.metadata.name}")
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()

    # validate that the parameters are as expected
    assert response_body["spec"]["params"] == expected_params


@pytest.mark.parametrize(
    "delete_api_version",
    [
        "v1",
        "v2",
    ],
)
def test_delete_project_not_found_in_leader(
    unversioned_client: TestClient,
    mock_project_follower_iguazio_client,
    mocked_k8s_helper,
    delete_api_version: str,
) -> None:
    archived_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="archived-project"),
        spec=mlrun.common.schemas.ProjectSpec(),
        status=mlrun.common.schemas.ProjectStatus(
            state=mlrun.common.schemas.ProjectState.archived
        ),
    )

    online_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="online-project"),
        spec=mlrun.common.schemas.ProjectSpec(),
    )

    response = unversioned_client.post("v1/projects", json=archived_project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(archived_project, response)

    response = unversioned_client.post("v1/projects", json=online_project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(online_project, response)

    with unittest.mock.patch.object(
        mock_project_follower_iguazio_client,
        "delete_project",
        side_effect=mlrun.errors.MLRunNotFoundError("Project not found"),
    ):
        response = unversioned_client.delete(
            f"{delete_api_version}/projects/{archived_project.metadata.name}",
        )
        assert response.status_code == HTTPStatus.ACCEPTED.value

        response = unversioned_client.get(
            f"v1/projects/{archived_project.metadata.name}",
        )
        assert response.status_code == HTTPStatus.NOT_FOUND.value

        response = unversioned_client.delete(
            f"{delete_api_version}/projects/{online_project.metadata.name}",
        )
        if response.status_code == HTTPStatus.ACCEPTED.value:
            assert delete_api_version == "v2"
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
                background_task.metadata.name
            )
            assert (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            )
            assert (
                "Failed to delete project online-project. Project not found in leader, but it is not in archived state."
                in background_task.status.error
            )

        else:
            assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

        response = unversioned_client.get(
            f"v1/projects/{online_project.metadata.name}",
        )
        assert response.status_code == HTTPStatus.OK.value


# Test should not run more than a few seconds because we test that if the background task fails,
# the wrapper task fails fast
@pytest.mark.usefixtures("mock_process_model_monitoring_secret")
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "delete_api_version",
    [
        "v1",
        "v2",
    ],
)
def test_delete_project_fail_fast(
    unversioned_client: TestClient,
    mock_project_follower_iguazio_client,
    delete_api_version: str,
) -> None:
    # Set the igz version for the project leader mock
    # We only test igz version < 3.5.5 flow because from 3.5.5 iguazio waits for the inner background task to
    # finish so the wrapper task does not wait for the inner task
    mlrun.mlconf.igz_version = "3.5.4"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        spec=mlrun.common.schemas.ProjectSpec(),
    )

    response = unversioned_client.post("v1/projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project, response)

    with unittest.mock.patch(
        "server.api.crud.projects.Projects.delete_project_resources",
        side_effect=Exception("some error"),
    ):
        response = unversioned_client.delete(
            f"{delete_api_version}/projects/{project.metadata.name}",
            headers={
                mlrun.common.schemas.HeaderNames.deletion_strategy: mlrun.common.schemas.DeletionStrategy.cascading,
            },
        )
        if delete_api_version == "v1":
            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value
            assert (
                "Failed to delete project project-name: some error"
                in response.json()["detail"]
            )
        else:
            assert response.status_code == HTTPStatus.ACCEPTED.value
            background_task = mlrun.common.schemas.BackgroundTask(**response.json())
            background_task = server.api.utils.background_tasks.InternalBackgroundTasksHandler().get_background_task(
                background_task.metadata.name
            )
            assert (
                background_task.status.state
                == mlrun.common.schemas.BackgroundTaskState.failed
            )
            assert (
                "Failed to delete project project-name: some error"
                in background_task.status.error
            )


def test_project_image_builder_validation(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # test image builder input is validated though output is not

    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        spec=mlrun.common.schemas.ProjectSpec(
            build=mlrun.common.schemas.ImageBuilder()
        ),
    )

    # create project
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value

    project.spec.build.requirements = ["pandas", "numpy"]
    expected_requirements = ["pandas", "numpy"]

    # store project request to save the requirements
    response = client.put(f"projects/{project.metadata.name}", json=project.dict())
    assert response.status_code == HTTPStatus.OK.value

    # get project and validate the project
    response = client.get(f"projects/{project.metadata.name}")
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()
    assert response_body["spec"]["build"]["requirements"] == expected_requirements

    project.spec.build.requirements = {"corrupted": "value"}

    # store project request to save the parameters
    response = client.put(f"projects/{project.metadata.name}", json=project.dict())
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value
    assert (
        '{"detail":[{"loc":["body","spec","build","requirements"],'
        '"msg":"value is not a valid list","type":"type_error.list"}]}'
        in str(response.content.decode())
    )

    # bypass the validation
    corrupted_project_name = "corrupted_project"
    full_object = {
        "metadata": {"name": corrupted_project_name},
        "spec": {"build": {"requirements": {"corrupted": "value"}}},
    }

    project_record = Project(name=corrupted_project_name, full_object=full_object)
    db.add(project_record)
    db.commit()

    # get the corrupted project
    response = client.get(f"projects/{corrupted_project_name}")
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()

    # ensure corrupted requirements passed validation
    assert (
        response_body["spec"]["build"]["requirements"]
        == full_object["spec"]["build"]["requirements"]
    )

    # ensures list projects
    response = client.get("projects")
    assert response.status_code == HTTPStatus.OK.value
    response_body = response.json()
    projects = response_body["projects"]

    # ensure corrupted requirements passed validation
    assert len(projects) == 2
    for project in projects:
        if project["metadata"]["name"] == corrupted_project_name:
            assert (
                project["spec"]["build"]["requirements"]
                == full_object["spec"]["build"]["requirements"]
            )
            break
    else:
        pytest.fail(f"Project {corrupted_project_name} not found")


def _create_resources_of_all_kinds(
    db_session: Session,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    project: str,
):
    db = server.api.utils.singletons.db.get_db()
    # add labels to project
    project_schema = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(
            name=project, labels={"key": "value"}
        ),
        spec=mlrun.common.schemas.ProjectSpec(description="some desc"),
    )
    server.api.utils.singletons.project_member.get_project_member().store_project(
        db_session, project, project_schema
    )

    # Create several functions with several tags
    labels = {
        "name": "value",
        "name2": "value2",
    }
    function = {
        "bla": "blabla",
        "metadata": {"labels": labels},
        "spec": {"asd": "asdasd"},
        "status": {"bla": "blabla"},
    }
    function_names = ["function_name_1", "function_name_2", "function_name_3"]
    function_tags = ["some_tag", "some_tag2", "some_tag3"]
    for function_name in function_names:
        for function_tag in function_tags:
            # change spec a bit so different (un-tagged) versions will be created
            for index in range(3):
                function["spec"]["index"] = index
                db.store_function(
                    db_session,
                    function,
                    function_name,
                    project,
                    tag=function_tag,
                    versioned=True,
                )

    # Create several artifacts with several tags
    artifact_template = {
        "metadata": {"labels": labels},
        "spec": {},
        "kind": "artifact",
        "status": {"bla": "blabla"},
    }
    artifact_keys = ["artifact_key_1", "artifact_key_2", "artifact_key_3"]
    artifact_trees = ["some_tree", "some_tree2", "some_tree3"]
    artifact_tags = ["some-tag", "some-tag2", "some-tag3"]
    for artifact_key in artifact_keys:
        for artifact_tree in artifact_trees:
            for artifact_tag in artifact_tags:
                for artifact_iter in range(3):
                    artifact = copy.deepcopy(artifact_template)
                    artifact["metadata"]["iter"] = artifact_iter
                    artifact["metadata"]["tag"] = artifact_tag
                    artifact["metadata"]["tree"] = artifact_tree

                    # pass a copy of the artifact to the store function, otherwise the store function will change the
                    # original artifact
                    db.store_artifact(
                        db_session,
                        artifact_key,
                        artifact,
                        iter=artifact_iter,
                        tag=artifact_tag,
                        project=project,
                        producer_id=artifact_tree,
                    )

    # Create several runs
    run = {
        "bla": "blabla",
        "metadata": {"name": "run-name", "labels": labels},
        "status": {"bla": "blabla"},
    }
    run_uids = ["some_uid", "some_uid2", "some_uid3"]
    for run_uid in run_uids:
        for run_iter in range(3):
            db.store_run(db_session, run, run_uid, project, run_iter)

    # Create several notifications
    for run_uid in run_uids:
        notification = mlrun.model.Notification(
            kind="slack",
            when=["completed", "error"],
            name=f"test-notification-{run_uid}",
            message="test-message",
            condition="",
            severity="info",
            params={"some-param": "some-value"},
        )
        db.store_run_notifications(db_session, [notification], run_uid, project)

    # Create alert notifications
    notification = mlrun.model.Notification(
        kind="slack",
        when=["completed", "error"],
        name="test-alert-notification",
        message="test-message",
        condition="",
        severity="info",
        params={"some-param": "some-value"},
    )

    alert = mlrun.common.schemas.AlertConfig(
        project=project,
        name="test_alert",
        summary="oops",
        severity=mlrun.common.schemas.alert.AlertSeverity.HIGH,
        entities={
            "kind": mlrun.common.schemas.alert.EventEntityKind.MODEL_ENDPOINT_RESULT,
            "project": project,
            "ids": [1234],
        },
        trigger={"events": [mlrun.common.schemas.alert.EventKind.DATA_DRIFT_DETECTED]},
        notifications=[{"notification": notification.to_dict()}],
        reset_policy=mlrun.common.schemas.alert.ResetPolicy.MANUAL,
    )
    alert = db.store_alert(db_session, alert)
    db.store_alert_notifications(db_session, [notification], alert.id, project)

    # Create several logs
    log = b"some random log"
    log_uids = ["some_uid", "some_uid2", "some_uid3"]
    for log_uid in log_uids:
        server.api.crud.Logs().store_log(log, project, log_uid)

    # Create several schedule
    schedule = {
        "bla": "blabla",
        "status": {"bla": "blabla"},
    }
    schedule_cron_trigger = mlrun.common.schemas.ScheduleCronTrigger(year=1999)
    schedule_names = ["schedule_name_1", "schedule_name_2", "schedule_name_3"]
    for schedule_name in schedule_names:
        server.api.utils.singletons.scheduler.get_scheduler().create_schedule(
            db_session,
            mlrun.common.schemas.AuthInfo(),
            project,
            schedule_name,
            mlrun.common.schemas.ScheduleKinds.job,
            schedule,
            schedule_cron_trigger,
            labels,
        )

    # Create several feature sets with several tags
    labels = {
        mlrun_constants.MLRunInternalLabels.owner: "nobody",
    }
    feature_set = mlrun.common.schemas.FeatureSet(
        metadata=mlrun.common.schemas.ObjectMetadata(
            name="dummy", tag="latest", labels=labels
        ),
        spec=mlrun.common.schemas.FeatureSetSpec(
            entities=[
                mlrun.common.schemas.Entity(
                    name="ent1", value_type="str", labels={"label": "1"}
                )
            ],
            features=[
                mlrun.common.schemas.Feature(
                    name="feat1", value_type="str", labels={"label": "1"}
                )
            ],
        ),
        status={},
    )
    feature_set_names = ["feature_set_1", "feature_set_2", "feature_set_3"]
    feature_set_tags = ["some_tag", "some_tag2", "some_tag3"]
    for feature_set_name in feature_set_names:
        for feature_set_tag in feature_set_tags:
            # change spec a bit so different (un-tagged) versions will be created
            for index in range(3):
                feature_set.metadata.name = feature_set_name
                feature_set.metadata.tag = feature_set_tag
                feature_set.spec.index = index
                db.store_feature_set(db_session, project, feature_set_name, feature_set)

    feature_vector = mlrun.common.schemas.FeatureVector(
        metadata=mlrun.common.schemas.ObjectMetadata(
            name="dummy", tag="latest", labels=labels
        ),
        spec=mlrun.common.schemas.ObjectSpec(),
        status=mlrun.common.schemas.ObjectStatus(state="created"),
    )
    feature_vector_names = ["feature_vector_1", "feature_vector_2", "feature_vector_3"]
    feature_vector_tags = ["some_tag", "some_tag2", "some_tag3"]
    for feature_vector_name in feature_vector_names:
        for feature_vector_tag in feature_vector_tags:
            # change spec a bit so different (un-tagged) versions will be created
            for index in range(3):
                feature_vector.metadata.name = feature_vector_name
                feature_vector.metadata.tag = feature_vector_tag
                feature_vector.spec.index = index
                db.store_feature_vector(
                    db_session, project, feature_vector_name, feature_vector
                )

    secrets = {f"secret_{i}": "a secret" for i in range(5)}
    k8s_secrets_mock.store_project_secrets(project, secrets)
    db.store_background_task(
        db_session,
        name="task",
        project=project,
        state=mlrun.common.schemas.BackgroundTaskState.running,
    )

    ds_profile = mlrun.common.schemas.DatastoreProfile(
        name="datastore_test_profile_name",
        type="datastore_test_profile_type",
        object="datastore_test_profile_body",
        project=project,
    )
    # create a datasource profile
    db.store_datastore_profile(db_session, ds_profile)


def _assert_resources_in_project(
    db_session: Session,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    project_member_mode: str,
    project: str,
    assert_no_resources: bool = False,
) -> tuple[dict, dict]:
    object_type_records_count_map = {
        "Logs": _assert_logs_in_project(project, assert_no_resources),
        "Schedules": _assert_schedules_in_project(project, assert_no_resources),
    }

    secrets = (
        {} if assert_no_resources else {f"secret_{i}": "a secret" for i in range(5)}
    )
    assert k8s_secrets_mock.get_project_secret_data(project) == secrets

    return (
        _assert_db_resources_in_project(
            db_session, project_member_mode, project, assert_no_resources
        ),
        object_type_records_count_map,
    )


def _assert_schedules_in_project(
    project: str,
    assert_no_resources: bool = False,
) -> int:
    number_of_schedules = len(
        server.api.utils.singletons.scheduler.get_scheduler()._list_schedules_from_scheduler(
            project
        )
    )
    if assert_no_resources:
        assert number_of_schedules == 0
    else:
        assert number_of_schedules > 0
    return number_of_schedules


def _assert_logs_in_project(
    project: str,
    assert_no_resources: bool = False,
) -> int:
    logs_path = server.api.api.utils.project_logs_path(project)
    number_of_log_files = 0
    if logs_path.exists():
        number_of_log_files = len(
            [
                file
                for file in os.listdir(str(logs_path))
                if os.path.isfile(os.path.join(str(logs_path), file))
            ]
        )
    if assert_no_resources:
        assert number_of_log_files == 0
    else:
        assert number_of_log_files > 0
    return number_of_log_files


def _assert_db_resources_in_project(
    db_session: Session,
    project_member_mode: str,
    project: str,
    assert_no_resources: bool = False,
) -> dict:
    table_name_records_count_map = {}
    for cls in _classes:
        # User support is not really implemented or in use
        # Run tags support is not really implemented or in use
        # Hub sources is not a project-level table, and hence is not relevant here.
        # Version is not a project-level table, and hence is not relevant here.
        # Features and Entities are not directly linked to project since they are sub-entity of feature-sets
        # Logs are saved as files, the DB table is not really in use
        # in follower mode the DB project tables are irrelevant
        # alert_templates are not tied to project and are pre-populated anyway
        if (
            cls.__name__ == "User"
            or cls.__tablename__ == "runs_tags"
            or cls.__tablename__ == "hub_sources"
            or cls.__tablename__ == "data_versions"
            or cls.__name__ == "Feature"
            or cls.__name__ == "Entity"
            or cls.__name__ == "Artifact"
            or cls.__name__ == "Log"
            or (
                cls.__tablename__ == "projects_labels"
                and project_member_mode == "follower"
            )
            or (cls.__tablename__ == "projects" and project_member_mode == "follower")
            or cls.__tablename__ == "alert_states"
            or cls.__tablename__ == "alert_templates"
        ):
            continue
        number_of_cls_records = 0
        # Label doesn't have project attribute
        # Project (obviously) doesn't have project attribute
        if cls.__name__ != "Label" and cls.__name__ != "Project":
            if (
                (
                    # Artifact table is deprecated, we are using ArtifactV2 instead
                    cls.__name__ == "Tag" and cls.__tablename__ == "artifacts_tags"
                )
                or (
                    # PaginationCache is not a project-level table
                    cls.__name__ == "PaginationCache"
                )
                or (
                    # Although project summaries are related to projects, their lifecycle is related
                    # to the project summary calculation cycle and not to the creation/deletion of projects
                    # (In each cycle the table is wiped clean and re-populated with only the existing projects)
                    cls.__name__ == "ProjectSummary"
                )
                or (
                    # TimeWindowTracker is not a project-level table
                    cls.__name__ == "TimeWindowTracker"
                )
            ):
                continue

            number_of_cls_records = (
                db_session.query(cls).filter_by(project=project).count()
            )
        elif cls.__name__ == "Label":
            if cls.__tablename__ == "functions_labels":
                number_of_cls_records = (
                    db_session.query(Function)
                    .join(cls)
                    .filter(Function.project == project)
                    .count()
                )
            if cls.__tablename__ == "runs_labels":
                number_of_cls_records = (
                    db_session.query(Run)
                    .join(cls)
                    .filter(Run.project == project)
                    .count()
                )
            if cls.__tablename__ == "artifacts_v2_labels":
                number_of_cls_records = (
                    db_session.query(ArtifactV2)
                    .join(cls)
                    .filter(ArtifactV2.project == project)
                    .count()
                )
            if cls.__tablename__ == "feature_sets_labels":
                number_of_cls_records = (
                    db_session.query(FeatureSet)
                    .join(cls)
                    .filter(FeatureSet.project == project)
                    .count()
                )
            if cls.__tablename__ == "features_labels":
                number_of_cls_records = (
                    db_session.query(FeatureSet)
                    .join(Feature)
                    .join(cls)
                    .filter(FeatureSet.project == project)
                    .count()
                )
            if cls.__tablename__ == "entities_labels":
                number_of_cls_records = (
                    db_session.query(FeatureSet)
                    .join(Entity)
                    .join(cls)
                    .filter(FeatureSet.project == project)
                    .count()
                )
            if cls.__tablename__ == "schedules_v2_labels":
                number_of_cls_records = (
                    db_session.query(Schedule)
                    .join(cls)
                    .filter(Schedule.project == project)
                    .count()
                )
            if cls.__tablename__ == "feature_vectors_labels":
                number_of_cls_records = (
                    db_session.query(FeatureVector)
                    .join(cls)
                    .filter(FeatureVector.project == project)
                    .count()
                )
            if cls.__tablename__ == "projects_labels":
                number_of_cls_records = (
                    db_session.query(Project)
                    .join(cls)
                    .filter(Project.name == project)
                    .count()
                )
            if cls.__tablename__ == "artifacts_labels":
                # Artifact table is deprecated, we are using ArtifactV2 instead
                continue
        elif cls.__name__ == "Project":
            number_of_cls_records = (
                db_session.query(Project).filter(Project.name == project).count()
            )
        else:
            raise NotImplementedError(
                "You excluded an object from the regular handling but forgot to add special handling"
            )
        if assert_no_resources:
            assert (
                number_of_cls_records == 0
            ), f"Table {cls.__tablename__} records were found"
        else:
            assert (
                number_of_cls_records > 0
            ), f"Table {cls.__tablename__} records were not found"
        table_name_records_count_map[cls.__tablename__] = number_of_cls_records
    return table_name_records_count_map


def _list_project_names_and_assert(
    client: TestClient, expected_names: list[str], params: dict = None
):
    params = params or {}
    params["format"] = mlrun.common.formatters.ProjectFormat.name_only
    # list - names only - filter by state
    response = client.get(
        "projects",
        params=params,
    )
    assert (
        deepdiff.DeepDiff(
            expected_names,
            response.json()["projects"],
            ignore_order=True,
        )
        == {}
    )


def _assert_project_response(
    expected_project: mlrun.common.schemas.Project, response, extra_exclude: dict = None
):
    project = mlrun.common.schemas.Project(**response.json())
    _assert_project(expected_project, project, extra_exclude)


def _assert_project_summary(
    project_summary: mlrun.common.schemas.ProjectSummary,
    files_count: int,
    feature_sets_count: int,
    models_count: int,
    runs_completed_recent_count,
    runs_failed_recent_count: int,
    runs_running_count: int,
    schedules_count: int,
    distinct_scheduled_jobs_pending_count: int,
    distinct_scheduled_pipelines_pending_count: int,
    pipelines_running_count: int,
):
    assert project_summary.files_count == files_count
    assert project_summary.feature_sets_count == feature_sets_count
    assert project_summary.models_count == models_count
    assert project_summary.runs_completed_recent_count == runs_completed_recent_count
    assert project_summary.runs_failed_recent_count == runs_failed_recent_count
    assert project_summary.runs_running_count == runs_running_count
    assert project_summary.distinct_schedules_count == schedules_count
    assert (
        project_summary.distinct_scheduled_jobs_pending_count
        == distinct_scheduled_jobs_pending_count
    )
    assert (
        project_summary.distinct_scheduled_pipelines_pending_count
        == distinct_scheduled_pipelines_pending_count
    )
    assert project_summary.pipelines_running_count == pipelines_running_count


def _assert_project(
    expected_project: mlrun.common.schemas.Project,
    project: mlrun.common.schemas.Project,
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


def _create_artifacts(client: TestClient, project_name, artifacts_count, kind):
    for index in range(artifacts_count):
        key = f"{kind}-name-{index}"
        # create several versions of the same artifact to verify we're not counting all versions, just all artifacts
        # (unique key)
        for _ in range(3):
            uid = str(uuid4())
            artifact = {
                "kind": kind,
                "metadata": {"key": key, "project": project_name},
                "spec": {"src_path": "/some/local/path"},
            }
            response = client.post(
                f"projects/{project_name}/artifacts/{uid}/{key}", json=artifact
            )
            assert response.status_code == HTTPStatus.OK.value, response.json()


def _create_feature_sets(client: TestClient, project_name, feature_sets_count):
    for index in range(feature_sets_count):
        feature_set_name = f"feature-set-name-{index}"
        # create several versions of the same feature set to verify we're not counting all versions, just all feature
        # sets (unique name)
        for _ in range(3):
            feature_set = {
                "metadata": {"name": feature_set_name, "project": project_name},
                "spec": {"entities": [], "features": [], "some_field": str(uuid4())},
                "status": {},
            }
            response = client.post(
                f"projects/{project_name}/feature-sets", json=feature_set
            )
            assert response.status_code == HTTPStatus.OK.value, response.json()


def _create_functions(client: TestClient, project_name, functions_count):
    for index in range(functions_count):
        function_name = f"function-name-{index}"
        # create several versions of the same function to verify we're not counting all versions, just all functions
        # (unique name)
        for _ in range(3):
            function = {
                "metadata": {"name": function_name, "project": project_name},
                "spec": {"some_field": str(uuid4())},
            }
            response = client.post(
                FUNCTIONS_API.format(project=project_name, name=function_name),
                json=function,
                params={"versioned": True},
            )
            assert response.status_code == HTTPStatus.OK.value, response.json()


def _create_runs(
    client: TestClient, project_name, runs_count, state=None, start_time=None
):
    for index in range(runs_count):
        run_name = f"run-name-{str(uuid4())}"
        # create several runs of the same name to verify we're not counting all instances, just all unique run names
        for _ in range(3):
            run_uid = str(uuid4())
            run = {
                "kind": mlrun.artifacts.model.ModelArtifact.kind,
                "metadata": {
                    "name": run_name,
                    "uid": run_uid,
                    "project": project_name,
                },
            }
            if state:
                run["status"] = {
                    "state": state,
                }
            if start_time:
                run.setdefault("status", {})["start_time"] = start_time.isoformat()
            response = client.post(f"run/{project_name}/{run_uid}", json=run)
            assert response.status_code == HTTPStatus.OK.value, response.json()


def _create_schedule(
    client: TestClient,
    project_name,
    cron_trigger: mlrun.common.schemas.ScheduleCronTrigger,
    labels: dict = None,
):
    if not labels:
        labels = {}

    schedule_name = f"schedule-name-{str(uuid4())}"
    schedule = mlrun.common.schemas.ScheduleInput(
        name=schedule_name,
        kind=mlrun.common.schemas.ScheduleKinds.job,
        scheduled_object={"metadata": {"name": "something"}},
        cron_trigger=cron_trigger,
        labels=labels,
    )
    response = client.post(f"projects/{project_name}/schedules", json=schedule.dict())
    assert response.status_code == HTTPStatus.CREATED.value, response.json()


def _create_schedules(client: TestClient, project_name):
    schedules_count = 3
    distinct_scheduled_jobs_pending_count = 5
    distinct_scheduled_pipelines_pending_count = 7

    for _ in range(schedules_count):
        _create_schedule(
            client, project_name, mlrun.common.schemas.ScheduleCronTrigger(year=1999)
        )

    for _ in range(distinct_scheduled_jobs_pending_count):
        _create_schedule(
            client,
            project_name,
            mlrun.common.schemas.ScheduleCronTrigger(minute=10),
            {"kind": "job"},
        )

    for _ in range(distinct_scheduled_pipelines_pending_count):
        _create_schedule(
            client,
            project_name,
            mlrun.common.schemas.ScheduleCronTrigger(minute=10),
            {mlrun_constants.MLRunInternalLabels.workflow: "workflow"},
        )
    return (
        schedules_count
        + distinct_scheduled_jobs_pending_count
        + distinct_scheduled_pipelines_pending_count,
        distinct_scheduled_jobs_pending_count,
        distinct_scheduled_pipelines_pending_count,
    )


def _mock_pipelines(project_name):
    mlrun.mlconf.kfp_url = "http://some-random-url:8888"
    status_count_map = {
        mlrun_pipelines.common.models.RunStatuses.running: 4,
        mlrun_pipelines.common.models.RunStatuses.succeeded: 3,
        mlrun_pipelines.common.models.RunStatuses.failed: 2,
    }
    pipelines = []
    for status, count in status_count_map.items():
        for index in range(count):
            pipelines.append({"status": status, "project": project_name})

    def list_pipelines_return_value(*args, **kwargs):
        next_page_token = "some-token"
        if kwargs["page_token"] == "":
            return None, next_page_token, pipelines[: len(pipelines) // 2]
        elif kwargs["page_token"] == next_page_token:
            return None, None, pipelines[len(pipelines) // 2 :]

    server.api.crud.Pipelines().list_pipelines = unittest.mock.Mock(
        side_effect=list_pipelines_return_value
    )
    return status_count_map[mlrun_pipelines.common.models.RunStatuses.running]


def _create_project(client: TestClient, name: str):
    """Helper to create a project."""
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name),
        spec=mlrun.common.schemas.ProjectSpec(),
    )
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project, response)
    return project
