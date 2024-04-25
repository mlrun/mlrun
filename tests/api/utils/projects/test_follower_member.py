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
import typing
import unittest.mock

import deepdiff
import kfp
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import server.api.crud
import server.api.utils.background_tasks
import server.api.utils.projects.follower
import server.api.utils.projects.remotes.leader
import server.api.utils.singletons.db
import server.api.utils.singletons.project_member
import tests.api.conftest
from mlrun.utils import logger


@pytest.fixture()
async def projects_follower() -> (
    typing.Generator[server.api.utils.projects.follower.Member, None, None]
):
    logger.info("Creating projects follower")
    mlrun.mlconf.httpdb.projects.leader = "nop"
    mlrun.mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"
    server.api.utils.singletons.project_member.initialize_project_member()
    projects_follower = server.api.utils.singletons.project_member.get_project_member()
    yield projects_follower
    logger.info("Stopping projects follower")
    projects_follower.shutdown()


@pytest.fixture()
async def nop_leader(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
) -> server.api.utils.projects.remotes.leader.Member:
    projects_follower._leader_client.db_session = db
    return projects_follower._leader_client


def test_sync_projects(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
    monkeypatch,
):
    project_nothing_changed = _generate_project(name="project-nothing-changed")
    project_in_creation = _generate_project(
        name="project-in-creation", state=mlrun.common.schemas.ProjectState.creating
    )
    project_in_deletion = _generate_project(
        name="project-in-deletion", state=mlrun.common.schemas.ProjectState.deleting
    )
    project_will_be_in_deleting = _generate_project(
        name="project-will-be-in-deleting",
        state=mlrun.common.schemas.ProjectState.creating,
    )
    project_moved_to_deletion = _generate_project(
        name=project_will_be_in_deleting.metadata.name,
        state=mlrun.common.schemas.ProjectState.deleting,
    )
    project_will_be_offline = _generate_project(
        name="project-will-be-offline", state=mlrun.common.schemas.ProjectState.online
    )
    project_offline = _generate_project(
        name=project_will_be_offline.metadata.name,
        state=mlrun.common.schemas.ProjectState.offline,
    )
    project_only_in_db = _generate_project(name="only-in-db")
    project_will_be_unarchived = _generate_project(
        name="project-will-be-unarchived",
        state=mlrun.common.schemas.ProjectState.archived,
    )

    # check race condition with background task where sync might want to create a project that is just deleted
    project_just_deleted = _generate_project(name="project-just-deleted")
    monkeypatch.setattr(
        server.api.utils.background_tasks.InternalBackgroundTasksHandler,
        "get_active_background_task_by_kind",
        lambda _, kind, raise_on_not_found: project_just_deleted.metadata.name in kind,
    )

    for _project in [
        project_nothing_changed,
        project_in_creation,
        project_will_be_offline,
        project_only_in_db,
        project_will_be_in_deleting,
        project_will_be_unarchived,
    ]:
        projects_follower.create_project(db, _project)

    project_will_be_unarchived.status.state = mlrun.common.schemas.ProjectState.online
    nop_leader_list_projects_mock = unittest.mock.Mock(
        return_value=(
            [
                project_nothing_changed,
                project_in_creation,
                project_in_deletion,
                project_offline,
                project_moved_to_deletion,
                project_will_be_unarchived,
                project_just_deleted,
            ],
            None,
        )
    )
    nop_leader.list_projects = nop_leader_list_projects_mock
    projects_follower._sync_projects()
    _assert_list_projects(
        db,
        projects_follower,
        [
            project_nothing_changed,
            project_in_creation,
            project_offline,
            project_only_in_db,
            project_moved_to_deletion,
            project_will_be_unarchived,
        ],
    )

    # ensure after full sync project that is not in leader is archived
    project_only_in_db.status.state = mlrun.common.schemas.ProjectState.archived
    projects_follower._sync_projects(full_sync=True)
    _assert_list_projects(
        db,
        projects_follower,
        [
            project_nothing_changed,
            project_in_creation,
            project_offline,
            project_only_in_db,
            project_moved_to_deletion,
            project_will_be_unarchived,
        ],
    )


def test_create_project(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project()
    created_project, _ = projects_follower.create_project(
        db,
        project,
    )
    _assert_projects_equal(project, created_project)
    _assert_project_in_follower(db, projects_follower, project)


def test_store_project(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project()

    # project doesn't exist - store will create
    created_project, _ = projects_follower.store_project(
        db,
        project.metadata.name,
        project,
    )
    _assert_projects_equal(project, created_project)
    _assert_project_in_follower(db, projects_follower, project)

    project_update = _generate_project(description="new description")
    # project exists - store will update
    updated_project, _ = projects_follower.store_project(
        db,
        project.metadata.name,
        project_update,
    )
    _assert_projects_equal(project_update, updated_project)
    _assert_project_in_follower(db, projects_follower, project_update)


def test_patch_project(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project()

    # project doesn't exist - store will create
    created_project, _ = projects_follower.store_project(
        db,
        project.metadata.name,
        project,
    )
    _assert_projects_equal(project, created_project)
    _assert_project_in_follower(db, projects_follower, project)

    patched_description = "new description"
    patched_project, _ = projects_follower.patch_project(
        db, project.metadata.name, {"spec": {"description": patched_description}}
    )
    expected_patched_project = _generate_project(description=patched_description)
    expected_patched_project.status.state = mlrun.common.schemas.ProjectState.online
    _assert_projects_equal(expected_patched_project, patched_project)
    _assert_project_in_follower(db, projects_follower, expected_patched_project)


def test_delete_project(
    db: sqlalchemy.orm.Session,
    # k8s_secrets_mock fixture uses the client fixture which intializes the project member so must be declared
    # before the projects follower
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project()
    projects_follower.create_project(
        db,
        project,
    )
    _assert_project_in_follower(db, projects_follower, project)
    server.api.utils.singletons.db.get_db().verify_project_has_no_related_resources = (
        unittest.mock.Mock(return_value=None)
    )
    projects_follower.delete_project(
        db,
        project.metadata.name,
    )
    _assert_project_not_in_follower(db, projects_follower, project.metadata.name)

    # make sure another delete doesn't fail
    projects_follower.delete_project(
        db,
        project.metadata.name,
    )


def test_get_project(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project()
    projects_follower.create_project(
        db,
        project,
    )
    # this functions uses get_project to assert, second assert will verify we're raising not found error
    _assert_project_in_follower(db, projects_follower, project)
    _assert_project_not_in_follower(db, projects_follower, "name-doesnt-exist")


def test_get_project_owner(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    owner = "some-username"
    owner_access_key = "some-access-key"
    nop_leader.project_owner_access_key = owner_access_key
    project = _generate_project(owner=owner)
    projects_follower.create_project(
        db,
        project,
    )
    project_owner = projects_follower.get_project_owner(db, project.metadata.name)
    assert project_owner.username == owner
    assert project_owner.access_key == owner_access_key


def test_list_project(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    owner = "project-owner"
    project = _generate_project(name="name-1", owner=owner)
    archived_project = _generate_project(
        name="name-2",
        desired_state=mlrun.common.schemas.ProjectDesiredState.archived,
        state=mlrun.common.schemas.ProjectState.archived,
        owner=owner,
    )
    label_key = "key"
    label_value = "value"
    labeled_project = _generate_project(name="name-3", labels={label_key: label_value})
    archived_and_labeled_project = _generate_project(
        name="name-4",
        desired_state=mlrun.common.schemas.ProjectDesiredState.archived,
        state=mlrun.common.schemas.ProjectState.archived,
        labels={label_key: label_value},
    )
    all_projects = {
        _project.metadata.name: _project
        for _project in [
            project,
            archived_project,
            labeled_project,
            archived_and_labeled_project,
        ]
    }
    for _project in all_projects.values():
        projects_follower.create_project(
            db,
            _project,
        )
    # list all
    _assert_list_projects(db, projects_follower, list(all_projects.values()))

    # list archived
    _assert_list_projects(
        db,
        projects_follower,
        [archived_project, archived_and_labeled_project],
        state=mlrun.common.schemas.ProjectState.archived,
    )

    # list by owner
    _assert_list_projects(
        db,
        projects_follower,
        [project, archived_project],
        owner=owner,
    )

    # list specific names only
    _assert_list_projects(
        db,
        projects_follower,
        [archived_project, labeled_project],
        names=[archived_project.metadata.name, labeled_project.metadata.name],
    )

    # list no valid names
    _assert_list_projects(
        db,
        projects_follower,
        [],
        names=[],
    )

    # list labeled - key existence
    _assert_list_projects(
        db,
        projects_follower,
        [labeled_project, archived_and_labeled_project],
        labels=[label_key],
    )

    # list labeled - key value match
    _assert_list_projects(
        db,
        projects_follower,
        [labeled_project, archived_and_labeled_project],
        labels=[f"{label_key}={label_value}"],
    )

    # list labeled - key value match and key existence
    _assert_list_projects(
        db,
        projects_follower,
        [labeled_project, archived_and_labeled_project],
        labels=[f"{label_key}={label_value}", label_key],
    )

    # list labeled - key value match and key existence
    _assert_list_projects(
        db,
        projects_follower,
        [labeled_project, archived_and_labeled_project],
        labels=[f"{label_key}={label_value}", label_key],
    )

    # list labeled and archived - key value match and key existence
    _assert_list_projects(
        db,
        projects_follower,
        [archived_and_labeled_project],
        state=mlrun.common.schemas.ProjectState.archived,
        labels=[f"{label_key}={label_value}", label_key],
    )


@pytest.mark.asyncio
async def test_list_project_summaries(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project(name="name-1")
    project_summary = mlrun.common.schemas.ProjectSummary(
        name=project.metadata.name,
        files_count=4,
        feature_sets_count=5,
        models_count=6,
        runs_failed_recent_count=7,
        runs_running_count=8,
        schedules_count=1,
        pipelines_running_count=2,
    )
    server.api.crud.Projects().generate_projects_summaries = unittest.mock.Mock(
        return_value=asyncio.Future()
    )
    server.api.crud.Projects().generate_projects_summaries.return_value.set_result(
        [project_summary]
    )
    project_summaries = await projects_follower.list_project_summaries(db)
    assert len(project_summaries.project_summaries) == 1
    assert (
        deepdiff.DeepDiff(
            project_summaries.project_summaries[0].dict(),
            project_summary.dict(),
            ignore_order=True,
        )
        == {}
    )


@pytest.mark.asyncio
async def test_list_project_summaries_fails_to_list_pipeline_runs(
    kfp_client_mock: kfp.Client,
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project_name = "project-name"
    _generate_project(name=project_name)
    server.api.utils.singletons.db.get_db().list_projects = unittest.mock.Mock(
        return_value=mlrun.common.schemas.ProjectsOutput(projects=[project_name])
    )
    server.api.crud.projects.Projects()._list_pipelines = unittest.mock.Mock(
        side_effect=mlrun.errors.MLRunNotFoundError("not found")
    )

    server.api.utils.singletons.db.get_db().get_project_resources_counters = (
        unittest.mock.AsyncMock(return_value=tuple({project_name: i} for i in range(6)))
    )
    project_summaries = await projects_follower.list_project_summaries(db)
    assert len(project_summaries.project_summaries) == 1
    assert project_summaries.project_summaries[0].name == project_name
    assert project_summaries.project_summaries[0].pipelines_running_count is None
    assert project_summaries.project_summaries[0].files_count == 0


def test_list_project_leader_format(
    db: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    nop_leader: server.api.utils.projects.remotes.leader.Member,
):
    project = _generate_project(name="name-1")
    server.api.utils.singletons.db.get_db().list_projects = unittest.mock.Mock(
        return_value=mlrun.common.schemas.ProjectsOutput(projects=[project])
    )
    projects = projects_follower.list_projects(
        db,
        format_=mlrun.common.schemas.ProjectsFormat.leader,
        projects_role=mlrun.common.schemas.ProjectsRole.nop,
    )
    assert (
        deepdiff.DeepDiff(
            projects.projects[0].data,
            project.dict(),
            ignore_order=True,
        )
        == {}
    )


def _assert_list_projects(
    db_session: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    expected_projects: list[mlrun.common.schemas.Project],
    **kwargs,
):
    projects = projects_follower.list_projects(db_session, **kwargs)
    assert len(projects.projects) == len(expected_projects)
    expected_projects_map = {
        _project.metadata.name: _project for _project in expected_projects
    }
    for project in projects.projects:
        _assert_projects_equal(project, expected_projects_map[project.metadata.name])

    # assert again - with name only format
    projects = projects_follower.list_projects(
        db_session, format_=mlrun.common.schemas.ProjectsFormat.name_only, **kwargs
    )
    assert len(projects.projects) == len(expected_projects)
    assert (
        deepdiff.DeepDiff(
            projects.projects,
            list(expected_projects_map.keys()),
            ignore_order=True,
        )
        == {}
    )


def _generate_project(
    name="project-name",
    description="some description",
    desired_state=mlrun.common.schemas.ProjectDesiredState.online,
    state=mlrun.common.schemas.ProjectState.online,
    labels: typing.Optional[dict] = None,
    owner="some-owner",
):
    return mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=name, labels=labels),
        spec=mlrun.common.schemas.ProjectSpec(
            description=description,
            desired_state=desired_state,
            owner=owner,
        ),
        status=mlrun.common.schemas.ProjectStatus(
            state=state,
        ),
    )


def _assert_projects_equal(project_1, project_2):
    exclude = {"metadata": {"created"}, "status": {"state"}}
    assert (
        deepdiff.DeepDiff(
            project_1.dict(exclude=exclude),
            project_2.dict(exclude=exclude),
            ignore_order=True,
        )
        == {}
    )
    assert mlrun.common.schemas.ProjectState(
        project_1.status.state
    ) == mlrun.common.schemas.ProjectState(project_2.status.state)


def _assert_project_not_in_follower(
    db_session: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    project_name: str,
):
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        projects_follower.get_project(db_session, project_name)


def _assert_project_in_follower(
    db_session: sqlalchemy.orm.Session,
    projects_follower: server.api.utils.projects.follower.Member,
    project: mlrun.common.schemas.Project,
):
    follower_project = projects_follower.get_project(db_session, project.metadata.name)
    _assert_projects_equal(project, follower_project)
