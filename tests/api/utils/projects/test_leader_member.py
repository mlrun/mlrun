import typing
import unittest.mock

import pytest
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.member
import mlrun.api.utils.projects.leader
import mlrun.config
import mlrun.errors
from mlrun.utils import logger


@pytest.fixture()
async def projects_leader() -> typing.Generator[
    mlrun.api.utils.projects.leader.Member, None, None
]:
    logger.info("Creating projects leader")
    mlrun.config.config.httpdb.projects.leader = "nop"
    mlrun.config.config.httpdb.projects.followers = "nop,nop2"
    mlrun.config.config.httpdb.projects.periodic_sync_interval = "0 seconds"
    projects_leader = mlrun.api.utils.projects.leader.Member()
    projects_leader.initialize()
    yield projects_leader
    logger.info("Stopping projects manager")
    projects_leader.shutdown()


@pytest.fixture()
async def nop_follower(
    projects_leader: mlrun.api.utils.projects.leader.Member,
) -> mlrun.api.utils.projects.remotes.member.Member:
    return projects_leader._followers["nop"]


@pytest.fixture()
async def second_nop_follower(
    projects_leader: mlrun.api.utils.projects.leader.Member,
) -> mlrun.api.utils.projects.remotes.member.Member:
    return projects_leader._followers["nop2"]


@pytest.fixture()
async def leader_follower(
    projects_leader: mlrun.api.utils.projects.leader.Member,
) -> mlrun.api.utils.projects.remotes.member.Member:
    return projects_leader._leader_follower


def test_projects_sync_follower_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    second_nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(description=project_description),
    )
    nop_follower.create_project(
        None, project,
    )
    _assert_project_in_followers([nop_follower], project, enriched=False)
    _assert_no_projects_in_followers([leader_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )


def test_projects_sync_leader_project_syncing(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    second_nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(description=project_description),
    )
    leader_follower.create_project(None, project)
    invalid_project_name = "invalid_name"
    invalid_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=invalid_project_name),
    )
    leader_follower.create_project(
        None, invalid_project,
    )
    _assert_project_in_followers([leader_follower], project, enriched=False)
    _assert_project_in_followers([leader_follower], invalid_project, enriched=False)
    _assert_no_projects_in_followers([nop_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower], project
    )
    _assert_project_not_in_followers(
        [nop_follower, second_nop_follower], invalid_project_name,
    )


def test_projects_sync_multiple_follower_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    second_nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    second_follower_project_name = "project-name-2"
    second_follower_project_description = "some description 2"
    second_follower_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=second_follower_project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description=second_follower_project_description
        ),
    )
    both_followers_project_name = "project-name"
    both_followers_project_description = "some description"
    both_followers_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=both_followers_project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description=both_followers_project_description
        ),
    )
    nop_follower.create_project(
        None, both_followers_project,
    )
    second_nop_follower.create_project(
        None, both_followers_project,
    )
    second_nop_follower.create_project(
        None, second_follower_project,
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
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            description=project_description,
            desired_state=mlrun.api.schemas.ProjectState.archived,
        ),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_create_and_store_project_failure_invalid_name(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    cases = [
        {"name": "asd3", "valid": True},
        {"name": "asd-asd", "valid": True},
        {"name": "333", "valid": True},
        {"name": "3-a-b", "valid": True},
        {"name": "5-a-a-5", "valid": True},
        {
            # Invalid because the first letter is -
            "name": "-as-123-2-8a",
            "valid": False,
        },
        {
            # Invalid because there is .
            "name": "as-123-2.a",
            "valid": False,
        },
        {
            # Invalid because A is not allowed
            "name": "As-123-2-8Aa",
            "valid": False,
        },
        {
            # Invalid because _ is not allowed
            "name": "as-123_2-8aa",
            "valid": False,
        },
        {
            # Invalid because it's more than 63 characters
            "name": "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            "valid": False,
        },
    ]
    for case in cases:
        project_name = case["name"]
        project = mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        )
        if case["valid"]:
            projects_leader.create_project(
                None, project,
            )
            _assert_project_in_followers([leader_follower], project)
            projects_leader.store_project(
                None, project_name, project,
            )
            _assert_project_in_followers([leader_follower], project)
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                projects_leader.create_project(
                    None, project,
                )
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                projects_leader.store_project(
                    None, project_name, project,
                )
            _assert_project_not_in_followers([leader_follower], project_name)


def test_ensure_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    projects_leader.ensure_project(
        None, project_name,
    )
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # further calls should do nothing
    projects_leader.ensure_project(
        None, project_name,
    )
    projects_leader.ensure_project(
        None, project_name,
    )


def test_store_project_creation(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(description=project_description),
    )
    _assert_no_projects_in_followers([leader_follower, nop_follower])

    projects_leader.store_project(
        None, project_name, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_store_project_update(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(description=project_description),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # removing description from the projects and changing desired state
    updated_project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(
            desired_state=mlrun.api.schemas.ProjectState.archived
        ),
    )

    projects_leader.store_project(
        None, project_name, updated_project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], updated_project)


def test_patch_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project, enriched=False
    )

    # Adding description to the project and changing state
    project_description = "some description"
    project_desired_state = mlrun.api.schemas.ProjectState.archived
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
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.store_project(
            None,
            project_name,
            mlrun.api.schemas.Project(
                metadata=mlrun.api.schemas.ProjectMetadata(name="different-name"),
            ),
        )
    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.patch_project(
            None, project_name, {"metadata": {"name": "different-name"}},
        )
    _assert_project_in_followers([leader_follower, nop_follower], project)


def test_delete_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    projects_leader.delete_project(None, project_name)
    _assert_no_projects_in_followers([leader_follower, nop_follower])


def test_delete_project_follower_failure(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    def mock_failed_delete(*args, **kwargs):
        raise RuntimeError()

    project_name = "project-name"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    nop_follower.delete_project = mock_failed_delete

    with pytest.raises(RuntimeError):
        projects_leader.delete_project(None, project_name)

    # deletion from leader should happen only after successful deletion from followers so ensure project still in leader
    _assert_project_in_followers([leader_follower], project)


def test_list_projects(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # add some project to follower
    nop_follower.create_project(
        None,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name="some-other-project"),
        ),
    )

    # assert list considers only the leader
    projects = projects_leader.list_projects(None)
    assert len(projects.projects) == 1
    assert projects.projects[0].metadata.name == project_name


def test_get_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        spec=mlrun.api.schemas.ProjectSpec(description=project_description),
    )
    projects_leader.create_project(
        None, project,
    )
    _assert_project_in_followers([leader_follower, nop_follower], project)

    # change project description in follower
    nop_follower.patch_project(
        None, project_name, {"spec": {"description": "updated description"}},
    )

    # assert get considers only the leader
    project = projects_leader.get_project(None, project_name)
    assert project.metadata.name == project_name
    assert project.spec.description == project_description


def _assert_project_not_in_followers(followers, name):
    for follower in followers:
        assert name not in follower._projects


def _assert_no_projects_in_followers(followers):
    for follower in followers:
        assert follower._projects == {}


def _assert_project_in_followers(
    followers, project: mlrun.api.schemas.Project, enriched=True
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
