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

import typing
import unittest.mock

import aioresponses
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.config
import mlrun.errors
from mlrun.utils import logger

import framework.utils.auth.providers.opa
import framework.utils.auth.verifier
import framework.utils.projects.leader
import framework.utils.projects.remotes.follower
import framework.utils.singletons.project_member


@pytest.fixture()
def projects_leader() -> typing.Iterator[framework.utils.projects.leader.Member]:
    logger.info("Creating projects leader")
    mlrun.mlconf.httpdb.projects.leader = "nop-self-leader"
    mlrun.mlconf.httpdb.projects.followers = "nop,nop2"
    mlrun.mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"
    framework.utils.singletons.project_member.initialize_project_member()
    projects_leader = framework.utils.singletons.project_member.get_project_member()
    yield projects_leader
    logger.info("Stopping projects leader")
    projects_leader.shutdown()


@pytest.fixture()
def nop_follower(
    projects_leader: framework.utils.projects.leader.Member,
) -> framework.utils.projects.remotes.follower.Member:
    return projects_leader._followers["nop"]


@pytest.fixture()
def second_nop_follower(
    projects_leader: framework.utils.projects.leader.Member,
) -> framework.utils.projects.remotes.follower.Member:
    return projects_leader._followers["nop2"]


@pytest.fixture()
def leader_follower(
    projects_leader: framework.utils.projects.leader.Member,
) -> framework.utils.projects.remotes.follower.Member:
    return projects_leader._leader_follower


@pytest.mark.parametrize(
    "method_name",
    [
        "create_project",
        "store_project",
    ],
)
@pytest.mark.parametrize(
    "explicit_owner, auth_user, expected_owner",
    [
        (None, "auth-user", "auth-user"),
        ("explicit-owner", "auth-user", "explicit-owner"),
        ("explicit-owner", "explicit-owner", "explicit-owner"),
    ],
)
def test_project_owner_from_auth_info_only_when_missing(
    projects_leader: framework.utils.projects.leader.Member,
    method_name: str,
    explicit_owner: str | None,
    auth_user: str,
    expected_owner: str,
):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        spec=mlrun.common.schemas.ProjectSpec(
            description="some description",
            owner=explicit_owner,
        ),
    )
    auth_info = mlrun.common.schemas.AuthInfo(username=auth_user)

    if method_name == "create_project":
        created_project, _ = projects_leader.create_project(
            None,
            project,
            auth_info=auth_info,
        )
        project_out = created_project
    elif method_name == "store_project":
        stored_project, _ = projects_leader.store_project(
            None,
            project.metadata.name,
            project,
            auth_info=auth_info,
        )
        project_out = stored_project
    else:
        raise ValueError(f"Unexpected method name: {method_name}")

    assert project_out.spec.owner == expected_owner


def test_projects_sync_follower_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    second_nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    nop_follower.create_project(
        None,
        project,
    )
    _assert_project_in_followers([nop_follower], project, enriched=False)
    _assert_no_projects_in_followers([leader_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )


def test_projects_sync_mid_deletion(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    second_nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    """
    This reproduces a bug in which projects sync is running during project deletion
    The sync starts after the project was removed from followers, but before it was removed from the leader, meaning the
    sync will recognize the project is missing in the followers, and create it in them, so finally after the delete
    process ends, the project exists in the followers, and not in the leader, on the next sync, the project will be
    created back in the leader causing the project to practically not being deleted.
    """
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    projects_leader.create_project(db, project)
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )
    original_leader_follower_delete_project = leader_follower.delete_project

    def mock_sync_projects_mid_deletion(*args, **kwargs):
        projects_leader._sync_projects()
        original_leader_follower_delete_project(*args, **kwargs)

    leader_follower.delete_project = mock_sync_projects_mid_deletion
    projects_leader.delete_project(db, project_name)

    _assert_no_projects_in_followers(
        [leader_follower, nop_follower, second_nop_follower]
    )

    projects_leader._sync_projects()
    _assert_no_projects_in_followers(
        [leader_follower, nop_follower, second_nop_follower]
    )


def test_projects_sync_leader_project_syncing(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    second_nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    enriched_project = project.copy(deep=True)
    # simulate project enrichment
    enriched_project.status.state = enriched_project.spec.desired_state
    leader_follower.create_project(None, enriched_project)
    invalid_project_name = "invalid_name"
    invalid_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=invalid_project_name),
    )
    leader_follower.create_project(
        None,
        invalid_project,
    )
    _assert_project_in_followers([leader_follower], project, enriched=False)
    _assert_project_in_followers([leader_follower], invalid_project, enriched=False)
    _assert_no_projects_in_followers([nop_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )
    _assert_project_not_in_followers(
        [nop_follower, second_nop_follower],
        invalid_project_name,
    )


def test_projects_sync_multiple_follower_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    second_nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    second_follower_project_name = "project-name-2"
    second_follower_project_description = "some description 2"
    second_follower_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(
            name=second_follower_project_name
        ),
        spec=mlrun.common.schemas.ProjectSpec(
            description=second_follower_project_description
        ),
    )
    both_followers_project_name = "project-name"
    both_followers_project_description = "some description"
    both_followers_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=both_followers_project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description=both_followers_project_description
        ),
    )
    nop_follower.create_project(
        None,
        both_followers_project,
    )
    second_nop_follower.create_project(
        None,
        both_followers_project,
    )
    second_nop_follower.create_project(
        None,
        second_follower_project,
    )
    leader_follower.create_project = unittest.mock.Mock(
        wraps=leader_follower.create_project
    )
    _assert_project_in_followers(
        [nop_follower, second_nop_follower], both_followers_project, enriched=False
    )
    _assert_project_in_followers(
        [second_nop_follower], second_follower_project, enriched=False
    )
    _assert_no_projects_in_followers([leader_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], both_followers_project
    )

    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], second_follower_project
    )

    # assert not tried to create project in leader twice
    assert leader_follower.create_project.call_count == 2


def test_create_project(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            description=project_description,
            desired_state=mlrun.common.schemas.ProjectState.archived,
        ),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)


@pytest.mark.parametrize(
    "project_name, valid",
    [
        ("asd3", True),
        ("asd-asd", True),
        ("333", True),
        ("3-a-b", True),
        ("5-a-a-5", True),
        (
            # Invalid because the first letter is -
            "-as-123-2-8a",
            False,
        ),
        (
            # Invalid because there is .
            "as-123-2.a",
            False,
        ),
        (
            # Invalid because A is not allowed
            "As-123-2-8Aa",
            False,
        ),
        (
            # Invalid because _ is not allowed
            "as-123_2-8aa",
            False,
        ),
        (
            # Invalid because it's more than 63 characters
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            False,
        ),
    ],
)
def test_create_and_store_project_failure_invalid_name(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
    project_name,
    valid,
):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    if valid:
        projects_leader.create_project(
            None,
            project,
        )
        _assert_project_in_followers([leader_follower], project)
        projects_leader.store_project(
            None,
            project_name,
            project,
        )
        _assert_project_in_followers([leader_follower], project)
    else:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            projects_leader.create_project(
                None,
                project,
            )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            projects_leader.store_project(
                None,
                project_name,
                project,
            )
        _assert_project_not_in_followers([leader_follower], project_name)


def test_ensure_project(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        projects_leader.ensure_project(
            None,
            project_name,
        )

    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # further calls should do nothing
    projects_leader.ensure_project(
        None,
        project_name,
    )
    projects_leader.ensure_project(
        None,
        project_name,
    )


def test_store_project_creation(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    _assert_no_projects_in_followers([leader_follower, nop_follower])

    projects_leader.store_project(
        None,
        project_name,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_store_project_update(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # removing description from the projects and changing desired state
    updated_project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(
            desired_state=mlrun.common.schemas.ProjectState.archived
        ),
    )

    projects_leader.store_project(
        None,
        project_name,
        updated_project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], updated_project)


def test_patch_project(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project, enriched=False
    )

    # Adding description to the project and changing state
    project_description = "some description"
    project_desired_state = mlrun.common.schemas.ProjectState.archived
    projects_leader.patch_project(
        None,
        project_name,
        {
            "spec": {
                "description": project_description,
                "desired_state": project_desired_state,
            }
        },
    )
    project.spec.description = project_description
    project.spec.desired_state = project_desired_state
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_store_and_patch_project_failure_conflict_body_path_name(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.store_project(
            None,
            project_name,
            mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(name="different-name"),
            ),
        )
    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.patch_project(
            None,
            project_name,
            {"metadata": {"name": "different-name"}},
        )
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_delete_project(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        db,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    projects_leader.delete_project(db, project_name)
    _assert_no_projects_in_followers([leader_follower, nop_follower])


def test_delete_project_follower_failure(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    def mock_failed_delete(*args, **kwargs):
        raise RuntimeError()

    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    nop_follower.delete_project = mock_failed_delete

    with pytest.raises(RuntimeError):
        projects_leader.delete_project(None, project_name)

    # deletion from leader should happen only after successful deletion from followers so ensure project still in leader
    _assert_project_in_followers([leader_follower], project)


def test_delete_project_follower_explicit_order(
    db: sqlalchemy.orm.Session,
    monkeypatch: pytest.MonkeyPatch,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    second_nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    # _follower_operation_order defines an explicit order for delete_project.
    # Setting nop2 before nop and making nop2 fail proves it runs first:
    # nop and the leader should still have the project.
    def mock_failed_delete(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(
        projects_leader,
        "_follower_operation_order",
        {"delete_project": ["nop2", "nop"]},
    )

    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )

    second_nop_follower.delete_project = mock_failed_delete

    with pytest.raises(RuntimeError):
        projects_leader.delete_project(None, project_name)

    # nop2 runs first (explicit order) and fails, so nop and leader should be untouched
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_list_projects(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # add some project to follower
    nop_follower.create_project(
        None,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="some-other-project"),
        ),
    )

    # assert list considers only the leader
    projects = projects_leader.list_projects(None)
    assert len(projects.projects) == 1
    assert projects.projects[0].metadata.name == project_name


def test_get_project(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    nop_follower: framework.utils.projects.remotes.follower.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(description=project_description),
    )
    projects_leader.create_project(
        None,
        project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # change project description in follower
    nop_follower.patch_project(
        None,
        project_name,
        {"spec": {"description": "updated description"}},
    )

    # assert get considers only the leader
    project = projects_leader.get_project(None, project_name)
    assert project.metadata.name == project_name
    assert project.spec.description == project_description


@pytest.mark.asyncio
async def test_ensure_project_populates_opa_owner_cache_across_replicas(
    db: sqlalchemy.orm.Session,
    projects_leader: framework.utils.projects.leader.Member,
    leader_follower: framework.utils.projects.remotes.follower.Member,
):
    """
    Simulates the multi-replica OPA manifest propagation race

    1. "API A" creates a project (project stored in DB with owner)
    2. "API B" receives a follow-up request (e.g. set_secrets)
       - OPA sidecar on B has NO manifest (always denies)
       - ensure_project fetches project from DB, sees owner matches, populates local cache
       - query_permissions hits cache → bypasses OPA → succeeds

    We simulate "empty OPA manifest" by configuring OPA to always return {"result": false}.
    """
    project_name = "test-cross-replica-project"
    owner_username = "project-creator"
    owner_user_id = "user-id-123"

    # --- Step 1: "API A" creates the project (stored in shared DB with owner) ---
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.common.schemas.ProjectSpec(owner=owner_username),
    )
    auth_info = mlrun.common.schemas.AuthInfo(
        username=owner_username,
        user_id=owner_user_id,
    )
    projects_leader.create_project(None, project, auth_info=auth_info)

    # Verify project is in "DB" (nop follower) with owner set
    stored = leader_follower.get_project(None, project_name)
    assert stored.spec.owner == owner_username

    # --- Step 2: "API B" — fresh OPA provider, empty manifest (always denies) ---
    api_url = "http://127.0.0.1:8181"
    permission_query_path = "/v1/data/service/authz/allow"
    mlrun.mlconf.httpdb.authorization.opa.address = api_url
    mlrun.mlconf.httpdb.authorization.opa.permission_query_path = permission_query_path
    mlrun.mlconf.httpdb.authorization.opa.log_level = 10
    mlrun.mlconf.httpdb.authorization.mode = "opa"

    opa_provider = framework.utils.auth.providers.opa.Provider()
    opa_provider.__init__()

    # Verify OPA denies (no manifest) BEFORE ensure_project
    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            f"{api_url}{permission_query_path}",
            payload={"result": False},
        )
        with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
            await opa_provider.query_permissions(
                f"/resources/projects/{project_name}/functions",
                mlrun.common.schemas.AuthorizationAction.create,
                auth_info,
            )

    # --- Step 3: ensure_project on "API B" — fetches from DB, populates cache ---
    projects_leader.ensure_project(None, project_name, auth_info=auth_info)

    # --- Step 4: OPA query now succeeds via cache — no OPA call made ---
    with aioresponses.aioresponses() as aiohttp_mock:
        # OPA still configured to deny, but cache should bypass it entirely
        allowed = await opa_provider.query_permissions(
            f"/resources/projects/{project_name}/functions",
            mlrun.common.schemas.AuthorizationAction.create,
            auth_info,
        )
        assert allowed is True
        # OPA was never called — the cache handled it
        aiohttp_mock.assert_not_called()

    # --- Step 5: Verify non-owner is NOT cached (security check) ---
    non_owner_auth = mlrun.common.schemas.AuthInfo(
        username="other-user",
        user_id="other-user-id",
    )
    projects_leader.ensure_project(None, project_name, auth_info=non_owner_auth)

    with aioresponses.aioresponses() as aiohttp_mock:
        aiohttp_mock.post(
            f"{api_url}{permission_query_path}",
            payload={"result": False},
        )
        with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
            await opa_provider.query_permissions(
                f"/resources/projects/{project_name}/functions",
                mlrun.common.schemas.AuthorizationAction.create,
                non_owner_auth,
            )

    await opa_provider._sessions.async_close()


def _assert_project_not_in_followers(followers, name):
    for follower in followers:
        assert name not in follower._projects


def _assert_no_projects_in_followers(followers):
    for follower in followers:
        assert follower._projects == {}


def _assert_project_in_followers(
    followers, project: mlrun.common.schemas.Project, enriched=True
):
    for follower in followers:
        assert (
            follower._projects[project.metadata.name].metadata.name
            == project.metadata.name
        )
        assert (
            follower._projects[project.metadata.name].spec.description
            == project.spec.description
        )
        assert (
            follower._projects[project.metadata.name].spec.desired_state
            == project.spec.desired_state
        )
        if enriched:
            assert (
                follower._projects[project.metadata.name].status.state
                == project.spec.desired_state
            )
