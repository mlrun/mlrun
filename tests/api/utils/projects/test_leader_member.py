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
    nop_follower.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers([nop_follower], project_name, project_description)
    _assert_no_projects_in_followers([leader_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower],
        project_name,
        project_description,
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
    leader_follower.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers([leader_follower], project_name, project_description)
    _assert_no_projects_in_followers([nop_follower, second_nop_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower],
        project_name,
        project_description,
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
    both_followers_project_name = "project-name"
    both_followers_project_description = "some description"
    nop_follower.create_project(
        None,
        mlrun.api.schemas.Project(
            name=both_followers_project_name,
            description=both_followers_project_description,
        ),
    )
    second_nop_follower.create_project(
        None,
        mlrun.api.schemas.Project(
            name=both_followers_project_name,
            description=both_followers_project_description,
        ),
    )
    second_nop_follower.create_project(
        None,
        mlrun.api.schemas.Project(
            name=second_follower_project_name,
            description=second_follower_project_description,
        ),
    )
    leader_follower.create_project = unittest.mock.Mock(
        wraps=leader_follower.create_project
    )
    _assert_project_in_followers(
        [nop_follower, second_nop_follower],
        both_followers_project_name,
        both_followers_project_description,
    )
    _assert_project_in_followers(
        [second_nop_follower],
        second_follower_project_name,
        second_follower_project_description,
    )
    _assert_no_projects_in_followers([leader_follower])

    projects_leader._sync_projects()
    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower],
        both_followers_project_name,
        both_followers_project_description,
    )

    _assert_project_in_followers(
        [leader_follower, nop_follower, second_nop_follower],
        second_follower_project_name,
        second_follower_project_description,
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
    projects_leader.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project_name, project_description
    )


def test_create_and_store_project_failure_invalid_name(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    cases = [
        {"name": "asd3", "valid": True},
        {"name": "asd-asd", "valid": True},
        {"name": "333", "valid": True},
        {"name": "3.a-b", "valid": True},
        {"name": "5.a-a.5", "valid": True},
        {
            # Invalid because the first letter is -
            "name": "-as-123_2.8a",
            "valid": False,
        },
        {
            # Invalid because the last letter is .
            "name": "as-123_2.8a.",
            "valid": False,
        },
        {
            # Invalid because A is not allowed
            "name": "As-123_2.8Aa",
            "valid": False,
        },
        {
            # Invalid because _ is not allowed
            "name": "as-123_2.8Aa",
            "valid": False,
        },
        {
            # Invalid because it's more than 253 characters
            "name": "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-"
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-"
            "azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsxdcfvg-azsx",
            "valid": False,
        },
    ]
    for case in cases:
        project_name = case["name"]
        if case["valid"]:
            projects_leader.create_project(
                None, mlrun.api.schemas.Project(name=project_name),
            )
            _assert_project_in_followers([leader_follower], project_name)
            projects_leader.store_project(
                None, project_name, mlrun.api.schemas.Project(name=project_name),
            )
            _assert_project_in_followers([leader_follower], project_name)
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                projects_leader.create_project(
                    None, mlrun.api.schemas.Project(name=project_name),
                )
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                projects_leader.store_project(
                    None, project_name, mlrun.api.schemas.Project(name=project_name),
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
    _assert_project_in_followers([leader_follower, nop_follower], project_name)

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
    _assert_no_projects_in_followers([leader_follower, nop_follower])

    projects_leader.store_project(
        None,
        project_name,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project_name, project_description
    )


def test_store_project_update(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    projects_leader.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project_name, project_description
    )

    # removing description from the projects
    projects_leader.store_project(
        None, project_name, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)


def test_patch_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    projects_leader.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)

    # Adding description to the projects
    project_description = "some description"
    projects_leader.patch_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectPatch(description=project_description),
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project_name, project_description
    )


def test_store_and_patch_project_failure_conflict_body_path_name(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    projects_leader.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.store_project(
            None, project_name, mlrun.api.schemas.Project(name="different-name"),
        )
    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_leader.patch_project(
            None, project_name, mlrun.api.schemas.ProjectPatch(name="different-name"),
        )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)


def test_delete_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    projects_leader.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)

    projects_leader.delete_project(None, project_name)
    _assert_no_projects_in_followers([leader_follower, nop_follower])


def test_list_projects(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    projects_leader.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_followers([leader_follower, nop_follower], project_name)

    # add some project to follower
    nop_follower.create_project(
        None, mlrun.api.schemas.Project(name="some-other-project")
    )

    # assert list considers only the leader
    projects = projects_leader.list_projects(None)
    assert len(projects.projects) == 1
    assert projects.projects[0].name == project_name


def test_get_project(
    db: sqlalchemy.orm.Session,
    projects_leader: mlrun.api.utils.projects.leader.Member,
    nop_follower: mlrun.api.utils.projects.remotes.member.Member,
    leader_follower: mlrun.api.utils.projects.remotes.member.Member,
):
    project_name = "project-name"
    project_description = "some description"
    projects_leader.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_followers(
        [leader_follower, nop_follower], project_name, project_description
    )

    # change project description in follower
    nop_follower.patch_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectPatch(description="updated description"),
    )

    # assert list considers only the leader
    project = projects_leader.get_project(None, project_name)
    assert project.name == project_name
    assert project.description == project_description


def _assert_project_not_in_followers(followers, name):
    for follower in followers:
        assert name not in follower._projects


def _assert_no_projects_in_followers(followers):
    for follower in followers:
        assert follower._projects == {}


def _assert_project_in_followers(followers, name, description=None):
    for follower in followers:
        assert follower._projects[name].name == name
        assert follower._projects[name].description == description
