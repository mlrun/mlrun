import typing
import unittest.mock

import pytest
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.projects.manager
import mlrun.config
import mlrun.errors
from mlrun.utils import logger


@pytest.fixture()
async def projects_manager() -> typing.Generator[
    mlrun.api.utils.projects.manager.ProjectsManager, None, None
]:
    logger.info("Creating projects manager")
    mlrun.config.config.httpdb.projects.master_consumer = "nop"
    mlrun.config.config.httpdb.projects.consumers = "nop,nop2"
    mlrun.config.config.httpdb.projects.periodic_sync_interval = "0 seconds"
    projects_manager = mlrun.api.utils.projects.manager.ProjectsManager()
    projects_manager.initialize()
    yield projects_manager
    logger.info("Stopping projects manager")
    projects_manager.shutdown()


@pytest.fixture()
async def nop_consumer(
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
) -> mlrun.api.utils.projects.consumers.base.Consumer:
    return projects_manager._consumers["nop"]


@pytest.fixture()
async def second_nop_consumer(
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
) -> mlrun.api.utils.projects.consumers.base.Consumer:
    return projects_manager._consumers["nop2"]


@pytest.fixture()
async def nop_master(
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
) -> mlrun.api.utils.projects.consumers.base.Consumer:
    return projects_manager._master_consumer


def test_projects_sync_consumer_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    second_nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    nop_consumer.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers([nop_consumer], project_name, project_description)
    _assert_no_projects_in_consumers([nop_master, second_nop_consumer])

    projects_manager._sync_projects()
    _assert_project_in_consumers(
        [nop_master, nop_consumer, second_nop_consumer],
        project_name,
        project_description,
    )


def test_projects_sync_master_project_syncing(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    second_nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    nop_master.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers([nop_master], project_name, project_description)
    _assert_no_projects_in_consumers([nop_consumer, second_nop_consumer])

    projects_manager._sync_projects()
    _assert_project_in_consumers(
        [nop_master, nop_consumer, second_nop_consumer],
        project_name,
        project_description,
    )


def test_projects_sync_multiple_consumer_project_adoption(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    second_nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    second_consumer_project_name = "project-name-2"
    second_consumer_project_description = "some description 2"
    both_consumers_project_name = "project-name"
    both_consumers_project_description = "some description"
    nop_consumer.create_project(
        None,
        mlrun.api.schemas.Project(name=both_consumers_project_name, description=both_consumers_project_description),
    )
    second_nop_consumer.create_project(
        None,
        mlrun.api.schemas.Project(name=both_consumers_project_name, description=both_consumers_project_description),
    )
    second_nop_consumer.create_project(
        None,
        mlrun.api.schemas.Project(name=second_consumer_project_name, description=second_consumer_project_description),
    )
    nop_master.create_project = unittest.mock.Mock(wraps=nop_master.create_project)
    _assert_project_in_consumers(
        [nop_consumer, second_nop_consumer], both_consumers_project_name, both_consumers_project_description
    )
    _assert_project_in_consumers(
        [second_nop_consumer], second_consumer_project_name, second_consumer_project_description
    )
    _assert_no_projects_in_consumers([nop_master])

    projects_manager._sync_projects()
    _assert_project_in_consumers(
        [nop_master, nop_consumer, second_nop_consumer],
        both_consumers_project_name,
        both_consumers_project_description,
    )

    _assert_project_in_consumers(
        [nop_master, nop_consumer, second_nop_consumer],
        second_consumer_project_name,
        second_consumer_project_description,
    )

    # assert not tried to create project in master twice
    assert nop_master.create_project.call_count == 2


def test_create_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    projects_manager.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )


def test_ensure_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.ensure_project(
        None, project_name,
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    # further calls should do nothing
    projects_manager.ensure_project(
        None, project_name,
    )
    projects_manager.ensure_project(
        None, project_name,
    )


def test_store_project_creation(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    _assert_no_projects_in_consumers([nop_master, nop_consumer])

    projects_manager.store_project(
        None,
        project_name,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )


def test_store_project_update(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    projects_manager.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )

    # removing description from the projects
    projects_manager.store_project(
        None, project_name, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)


def test_patch_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    # Adding description to the projects
    project_description = "some description"
    projects_manager.patch_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectPatch(description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )


def test_store_and_patch_project_failure_conflict_body_path_name(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_manager.store_project(
            None, project_name, mlrun.api.schemas.Project(name="different-name"),
        )
    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_manager.patch_project(
            None, project_name, mlrun.api.schemas.ProjectPatch(name="different-name"),
        )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)


def test_delete_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    projects_manager.delete_project(None, project_name)
    _assert_no_projects_in_consumers([nop_master, nop_consumer])


def test_list_projects(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.Project(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    # add some project to consumer
    nop_consumer.create_project(
        None, mlrun.api.schemas.Project(name="some-other-project")
    )

    # assert list considers only the master
    projects = projects_manager.list_projects(None)
    assert len(projects.projects) == 1
    assert projects.projects[0].name == project_name


def test_get_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    projects_manager.create_project(
        None,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )

    # change project description in consumer
    nop_consumer.patch_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectPatch(description="updated description"),
    )

    # assert list considers only the master
    project = projects_manager.get_project(None, project_name)
    assert project.name == project_name
    assert project.description == project_description


def _assert_no_projects_in_consumers(consumers):
    for consumer in consumers:
        assert consumer._projects == {}


def _assert_project_in_consumers(consumers, name, description=None):
    for consumer in consumers:
        assert consumer._projects[name].name == name
        assert consumer._projects[name].description == description
