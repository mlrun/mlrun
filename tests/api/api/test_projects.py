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
import copy
import datetime
import http
import json.decoder
import os
import typing
import unittest.mock
from http import HTTPStatus
from uuid import uuid4

import deepdiff
import fastapi.testclient
import mergedeep
import pytest
import sqlalchemy.orm
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.api.utils
import mlrun.api.crud
import mlrun.api.main
import mlrun.api.schemas
import mlrun.api.utils.background_tasks
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.api.utils.singletons.logs_dir
import mlrun.api.utils.singletons.project_member
import mlrun.api.utils.singletons.scheduler
import mlrun.artifacts.dataset
import mlrun.artifacts.model
import mlrun.errors
import tests.api.conftest
from mlrun.api.db.sqldb.models import (
    Artifact,
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

ORIGINAL_VERSIONED_API_PREFIX = mlrun.api.main.BASE_VERSIONED_API_PREFIX


@pytest.fixture(params=["leader", "follower"])
def project_member_mode(request, db: Session) -> str:
    if request.param == "follower":
        mlrun.config.config.httpdb.projects.leader = "nop"
        mlrun.api.utils.singletons.project_member.initialize_project_member()
        mlrun.api.utils.singletons.project_member.get_project_member()._leader_client.db_session = (
            db
        )
    elif request.param == "leader":
        mlrun.config.config.httpdb.projects.leader = "mlrun"
        mlrun.api.utils.singletons.project_member.initialize_project_member()
    else:
        raise NotImplementedError(
            f"Provided project member mode is not supported. mode={request.param}"
        )
    yield request.param


def test_redirection_from_worker_to_chief_delete_project(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient, httpserver
):
    mlrun.mlconf.httpdb.clusterization.role = "worker"
    project = "test-project"
    endpoint = f"{ORIGINAL_VERSIONED_API_PREFIX}/projects/{project}"
    for strategy in mlrun.api.schemas.DeletionStrategy:
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
                        "reason": f"Project {project} can not be deleted since related resources found: x"
                    }
                },
            },
        ]:
            expected_status = test_case.get("expected_status")
            expected_response = test_case.get("expected_body")

            httpserver.expect_ordered_request(
                endpoint, method="DELETE"
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
    name1 = f"prj-{uuid4().hex}"
    project_1 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name1),
    )

    # create
    response = client.post("projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project_1, response)

    # create again
    response = client.post("projects", json=project_1.dict())
    assert response.status_code == HTTPStatus.CONFLICT.value


def test_get_non_existing_project(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    """
    At first we were doing auth before get - which caused get on non existing project to return unauthorized instead of
    not found - which "ruined" the `mlrun.get_or_create_project` logic - so adding a specific test to verify it works
    """
    project = "does-not-exist"
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions = (
        unittest.mock.AsyncMock(side_effect=mlrun.errors.MLRunUnauthorizedError("bla"))
    )
    response = client.get(f"projects/{project}")
    assert response.status_code == HTTPStatus.NOT_FOUND.value


def test_delete_project_with_resources(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
):
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
    response = client.delete(
        f"projects/{project_to_remove}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.check.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

    # deletion strategy - restricted - should fail because there are resources
    response = client.delete(
        f"projects/{project_to_remove}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

    # deletion strategy - cascading - should succeed and remove all related resources
    response = client.delete(
        f"projects/{project_to_remove}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.cascading.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

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
    response = client.delete(
        f"projects/{project_to_remove}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.check.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # deletion strategy - restricted - should succeed cause no project
    response = client.delete(
        f"projects/{project_to_remove}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value


def test_list_and_get_project_summaries(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    # create empty project
    empty_project_name = "empty-project"
    empty_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=empty_project_name),
    )
    response = client.post("projects", json=empty_project.dict())
    assert response.status_code == HTTPStatus.CREATED.value

    # create project with resources
    project_name = "project-with-resources"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value

    # create files for the project
    files_count = 5
    _create_artifacts(
        client, project_name, files_count, mlrun.artifacts.ChartArtifact.kind
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
        mlrun.runtimes.constants.RunStates.running,
    )

    # create completed runs for the project to make sure we're not mistakenly counting them
    _create_runs(client, project_name, 2, mlrun.runtimes.constants.RunStates.completed)

    # create failed runs for the project for less than 24 hours ago
    recent_failed_runs_count = 6
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    _create_runs(
        client,
        project_name,
        recent_failed_runs_count,
        mlrun.runtimes.constants.RunStates.error,
        one_hour_ago,
    )

    # create aborted runs for the project for less than 24 hours ago - make sure we count them as well
    recent_aborted_runs_count = 6
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    _create_runs(
        client,
        project_name,
        recent_failed_runs_count,
        mlrun.runtimes.constants.RunStates.aborted,
        one_hour_ago,
    )

    # create failed runs for the project for more than 24 hours ago to make sure we're not mistakenly counting them
    two_days_ago = datetime.datetime.now() - datetime.timedelta(hours=48)
    _create_runs(
        client, project_name, 3, mlrun.runtimes.constants.RunStates.error, two_days_ago
    )

    # create schedules for the project
    schedules_count = 3
    _create_schedules(
        client,
        project_name,
        schedules_count,
    )

    # mock pipelines for the project
    running_pipelines_count = _mock_pipelines(
        project_name,
    )

    # list project summaries
    response = client.get("project-summaries")
    project_summaries_output = mlrun.api.schemas.ProjectSummariesOutput(
        **response.json()
    )
    for index, project_summary in enumerate(project_summaries_output.project_summaries):
        if project_summary.name == empty_project_name:
            _assert_project_summary(project_summary, 0, 0, 0, 0, 0, 0, 0)
        elif project_summary.name == project_name:
            _assert_project_summary(
                project_summary,
                files_count,
                feature_sets_count,
                models_count,
                recent_failed_runs_count + recent_aborted_runs_count,
                running_runs_count,
                schedules_count,
                running_pipelines_count,
            )
        else:
            pytest.fail(f"Unexpected project summary returned: {project_summary}")

    # get project summary
    response = client.get(f"project-summaries/{project_name}")
    project_summary = mlrun.api.schemas.ProjectSummary(**response.json())
    _assert_project_summary(
        project_summary,
        files_count,
        feature_sets_count,
        models_count,
        recent_failed_runs_count + recent_aborted_runs_count,
        running_runs_count,
        schedules_count,
        running_pipelines_count,
    )


def test_list_project_summaries_different_installation_modes(
    db: Session, client: TestClient, project_member_mode: str
) -> None:
    """
    The list project summaries endpoint is used in our projects screen and tend to break in different installation modes
    """
    # create empty project
    empty_project_name = "empty-project"
    empty_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=empty_project_name),
    )
    response = client.post("projects", json=empty_project.dict())
    assert response.status_code == HTTPStatus.CREATED.value

    mlrun.api.crud.Pipelines().list_pipelines = unittest.mock.Mock(
        return_value=(0, None, [])
    )
    # Enterprise installation configuration post 3.4.0
    mlrun.mlconf.igz_version = "3.6.0-b26.20210904121245"
    mlrun.mlconf.kfp_url = "https://somekfp-url.com"
    mlrun.mlconf.namespace = "default-tenant"

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.api.schemas.ProjectSummariesOutput(
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
    )

    # Enterprise installation configuration pre 3.4.0
    mlrun.mlconf.igz_version = "3.2.0-b26.20210904121245"
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = "default-tenant"

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.api.schemas.ProjectSummariesOutput(
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
    )

    # Kubernetes installation configuration (mlrun-kit)
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = "mlrun"

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.api.schemas.ProjectSummariesOutput(
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
    )

    # Docker installation configuration
    mlrun.mlconf.igz_version = ""
    mlrun.mlconf.kfp_url = ""
    mlrun.mlconf.namespace = ""

    response = client.get("project-summaries")
    assert response.status_code == HTTPStatus.OK.value
    project_summaries_output = mlrun.api.schemas.ProjectSummariesOutput(
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
    )


def test_delete_project_deletion_strategy_check(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name="project-name"),
        spec=mlrun.api.schemas.ProjectSpec(),
    )

    # create
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project, response)

    # deletion strategy - check - should succeed because there are no resources
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.check.value
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
        f"func/{project.metadata.name}/{function_name}", json=function
    )
    assert response.status_code == HTTPStatus.OK.value

    # deletion strategy - check - should fail because there are resources
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.check.value
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

    response = client.get("funcs", params={"project": project_name})
    assert response.status_code == HTTPStatus.OK.value
    distinct_function_names = {
        function["metadata"]["name"] for function in response.json()["funcs"]
    }
    # ensure there are indeed several versions of the same function name
    assert len(distinct_function_names) < len(response.json()["funcs"])

    response = client.get("artifacts", params={"project": project_name, "tag": "*"})
    assert response.status_code == HTTPStatus.OK.value
    # ensure there are indeed several versions of the same artifact key
    distinct_artifact_keys = {
        (artifact["db_key"], artifact["iter"])
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

    mlrun.api.utils.singletons.db.get_db().delete_function = unittest.mock.Mock()
    mlrun.api.utils.singletons.db.get_db().del_artifact = unittest.mock.Mock()
    mlrun.api.utils.singletons.db.get_db().delete_feature_set = unittest.mock.Mock()
    mlrun.api.utils.singletons.db.get_db().delete_feature_vector = unittest.mock.Mock()
    # deletion strategy - check - should fail because there are resources
    response = client.delete(
        f"projects/{project_name}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.cascading.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    assert mlrun.api.utils.singletons.db.get_db().delete_function.call_count == len(
        distinct_function_names
    )
    assert mlrun.api.utils.singletons.db.get_db().del_artifact.call_count == len(
        distinct_artifact_keys
    )
    assert mlrun.api.utils.singletons.db.get_db().delete_feature_set.call_count == len(
        distinct_feature_set_names
    )
    assert (
        mlrun.api.utils.singletons.db.get_db().delete_feature_vector.call_count
        == len(distinct_feature_vector_names)
    )


def test_delete_project_deletion_strategy_check_external_resource(
    db: Session,
    client: TestClient,
    project_member_mode: str,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    mlrun.mlconf.namespace = "test-namespace"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name="project-name"),
        spec=mlrun.api.schemas.ProjectSpec(),
    )

    # create
    response = client.post("projects", json=project.dict())
    assert response.status_code == HTTPStatus.CREATED.value
    _assert_project_response(project, response)

    # Set a project secret
    k8s_secrets_mock.store_project_secrets("project-name", {"secret": "value"})

    # deletion strategy - check - should fail because there's a project secret
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value
    assert "project secrets" in response.text

    k8s_secrets_mock.delete_project_secrets("project-name", None)
    response = client.delete(
        f"projects/{project.metadata.name}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response


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
        project = mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        )
        mlrun.api.utils.singletons.db.get_db().create_project(db, project)
        project_names.append(project_name)

    # list in leader format
    response = client.get(
        "projects",
        params={"format": mlrun.api.schemas.ProjectsFormat.leader},
        headers={
            mlrun.api.schemas.HeaderNames.projects_role: mlrun.mlconf.httpdb.projects.leader
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
    project_1 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name1),
        spec=mlrun.api.schemas.ProjectSpec(
            description="banana", source="source", goals="some goals"
        ),
    )

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
            "desired_state": mlrun.api.schemas.ProjectState.archived,
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
    project_2 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=name2, labels=labels_2),
        spec=mlrun.api.schemas.ProjectSpec(description="banana2", source="source2"),
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
        "projects", params={"format": mlrun.api.schemas.ProjectsFormat.full}
    )
    projects_output = mlrun.api.schemas.ProjectsOutput(**response.json())
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
        client, [name1], params={"state": mlrun.api.schemas.ProjectState.archived}
    )

    # add function to project 1
    function_name = "function-name"
    function = {"metadata": {"name": function_name}}
    response = client.post(f"func/{name1}/{function_name}", json=function)
    assert response.status_code == HTTPStatus.OK.value

    # delete - restricted strategy, will fail because function exists
    response = client.delete(
        f"projects/{name1}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.restricted.value
        },
    )
    assert response.status_code == HTTPStatus.PRECONDITION_FAILED.value

    # delete - cascading strategy, will succeed and delete function
    response = client.delete(
        f"projects/{name1}",
        headers={
            mlrun.api.schemas.HeaderNames.deletion_strategy: mlrun.api.schemas.DeletionStrategy.cascading.value
        },
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    # ensure function is gone
    response = client.get(f"func/{name1}/{function_name}")
    assert response.status_code == HTTPStatus.NOT_FOUND.value

    # list
    _list_project_names_and_assert(client, [name2])


def _create_resources_of_all_kinds(
    db_session: Session,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    project: str,
):
    db = mlrun.api.utils.singletons.db.get_db()
    # add labels to project
    project_schema = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(
            name=project, labels={"key": "value"}
        ),
        spec=mlrun.api.schemas.ProjectSpec(description="some desc"),
    )
    mlrun.api.utils.singletons.project_member.get_project_member().store_project(
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
    artifact = {
        "bla": "blabla",
        "labels": labels,
        "status": {"bla": "blabla"},
    }
    artifact_keys = ["artifact_key_1", "artifact_key_2", "artifact_key_3"]
    artifact_uids = ["some_uid", "some_uid2", "some_uid3"]
    artifact_tags = ["some-tag", "some-tag2", "some-tag3"]
    for artifact_key in artifact_keys:
        for artifact_uid in artifact_uids:
            for artifact_tag in artifact_tags:
                for artifact_iter in range(3):
                    artifact["iter"] = artifact_iter
                    artifact["tag"] = artifact_tag
                    artifact["uid"] = artifact_uid
                    db.store_artifact(
                        db_session,
                        artifact_key,
                        artifact,
                        artifact_uid,
                        artifact_iter,
                        artifact_tag,
                        project,
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
        db.store_notifications(db_session, [notification], run_uid, project)

    # Create several logs
    log = b"some random log"
    log_uids = ["some_uid", "some_uid2", "some_uid3"]
    for log_uid in log_uids:
        mlrun.api.crud.Logs().store_log(log, project, log_uid)

    # Create several schedule
    schedule = {
        "bla": "blabla",
        "status": {"bla": "blabla"},
    }
    schedule_cron_trigger = mlrun.api.schemas.ScheduleCronTrigger(year=1999)
    schedule_names = ["schedule_name_1", "schedule_name_2", "schedule_name_3"]
    for schedule_name in schedule_names:
        mlrun.api.utils.singletons.scheduler.get_scheduler().create_schedule(
            db_session,
            mlrun.api.schemas.AuthInfo(),
            project,
            schedule_name,
            mlrun.api.schemas.ScheduleKinds.job,
            schedule,
            schedule_cron_trigger,
            labels,
        )

    # Create several feature sets with several tags
    labels = {
        "owner": "nobody",
    }
    feature_set = mlrun.api.schemas.FeatureSet(
        metadata=mlrun.api.schemas.ObjectMetadata(
            name="dummy", tag="latest", labels=labels
        ),
        spec=mlrun.api.schemas.FeatureSetSpec(
            entities=[
                mlrun.api.schemas.Entity(
                    name="ent1", value_type="str", labels={"label": "1"}
                )
            ],
            features=[
                mlrun.api.schemas.Feature(
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

    feature_vector = mlrun.api.schemas.FeatureVector(
        metadata=mlrun.api.schemas.ObjectMetadata(
            name="dummy", tag="latest", labels=labels
        ),
        spec=mlrun.api.schemas.ObjectSpec(),
        status=mlrun.api.schemas.ObjectStatus(state="created"),
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
        state=mlrun.api.schemas.BackgroundTaskState.running,
    )


def _assert_resources_in_project(
    db_session: Session,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    project_member_mode: str,
    project: str,
    assert_no_resources: bool = False,
) -> typing.Tuple[typing.Dict, typing.Dict]:
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
        mlrun.api.utils.singletons.scheduler.get_scheduler()._list_schedules_from_scheduler(
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
    logs_path = mlrun.api.api.utils.project_logs_path(project)
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
) -> typing.Dict:
    table_name_records_count_map = {}
    for cls in _classes:
        # User support is not really implemented or in use
        # Run tags support is not really implemented or in use
        # Marketplace sources is not a project-level table, and hence is not relevant here.
        # Version is not a project-level table, and hence is not relevant here.
        # Features and Entities are not directly linked to project since they are sub-entity of feature-sets
        # Logs are saved as files, the DB table is not really in use
        # in follower mode the DB project tables are irrelevant
        if (
            cls.__name__ == "User"
            or cls.__tablename__ == "runs_tags"
            or cls.__tablename__ == "marketplace_sources"
            or cls.__tablename__ == "data_versions"
            or cls.__name__ == "Feature"
            or cls.__name__ == "Entity"
            or cls.__name__ == "Log"
            or (
                cls.__tablename__ == "projects_labels"
                and project_member_mode == "follower"
            )
            or (cls.__tablename__ == "projects" and project_member_mode == "follower")
        ):
            continue
        number_of_cls_records = 0
        # Label doesn't have project attribute
        # Project (obviously) doesn't have project attribute
        if cls.__name__ != "Label" and cls.__name__ != "Project":
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
            if cls.__tablename__ == "artifacts_labels":
                number_of_cls_records = (
                    db_session.query(Artifact)
                    .join(cls)
                    .filter(Artifact.project == project)
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
    client: TestClient, expected_names: typing.List[str], params: typing.Dict = None
):
    params = params or {}
    params["format"] = mlrun.api.schemas.ProjectsFormat.name_only
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
    expected_project: mlrun.api.schemas.Project, response, extra_exclude: dict = None
):
    project = mlrun.api.schemas.Project(**response.json())
    _assert_project(expected_project, project, extra_exclude)


def _assert_project_summary(
    project_summary: mlrun.api.schemas.ProjectSummary,
    files_count: int,
    feature_sets_count: int,
    models_count: int,
    runs_failed_recent_count: int,
    runs_running_count: int,
    schedules_count: int,
    pipelines_running_count: int,
):
    assert project_summary.files_count == files_count
    assert project_summary.feature_sets_count == feature_sets_count
    assert project_summary.models_count == models_count
    assert project_summary.runs_failed_recent_count == runs_failed_recent_count
    assert project_summary.runs_running_count == runs_running_count
    assert project_summary.schedules_count == schedules_count
    assert project_summary.pipelines_running_count == pipelines_running_count


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
                f"artifact/{project_name}/{uid}/{key}", json=artifact
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
                f"func/{project_name}/{function_name}",
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


def _create_schedules(client: TestClient, project_name, schedules_count):
    for index in range(schedules_count):
        schedule_name = f"schedule-name-{str(uuid4())}"
        schedule = mlrun.api.schemas.ScheduleInput(
            name=schedule_name,
            kind=mlrun.api.schemas.ScheduleKinds.job,
            scheduled_object={"metadata": {"name": "something"}},
            cron_trigger=mlrun.api.schemas.ScheduleCronTrigger(year=1999),
        )
        response = client.post(
            f"projects/{project_name}/schedules", json=schedule.dict()
        )
        assert response.status_code == HTTPStatus.CREATED.value, response.json()


def _mock_pipelines(project_name):
    mlrun.mlconf.kfp_url = "http://some-random-url:8888"
    status_count_map = {
        mlrun.run.RunStatuses.running: 4,
        mlrun.run.RunStatuses.succeeded: 3,
        mlrun.run.RunStatuses.failed: 2,
    }
    pipelines = []
    for status, count in status_count_map.items():
        for index in range(count):
            pipelines.append({"status": status, "project": project_name})
    mlrun.api.crud.Pipelines().list_pipelines = unittest.mock.Mock(
        return_value=(None, None, pipelines)
    )
    return status_count_map[mlrun.run.RunStatuses.running]
