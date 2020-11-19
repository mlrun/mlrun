import typing

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
    projects_manager.start()
    yield projects_manager
    logger.info("Stopping projects manager")
    projects_manager.stop()


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
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )
    _assert_project_in_consumers([nop_consumer], project_name, project_description)
    _assert_no_projects_in_consumers([nop_master, second_nop_consumer])

    projects_manager._sync_projects()
    _assert_project_in_consumers(
        [nop_master, nop_consumer, second_nop_consumer],
        project_name,
        project_description,
    )


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
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
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


def test_update_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.ProjectCreate(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    # Adding description to the projects
    project_description = "some description"
    projects_manager.update_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectUpdate(description=project_description),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )


def test_update_project_failure_conflict_body_path_name(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.ProjectCreate(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    with pytest.raises(mlrun.errors.MLRunConflictError):
        projects_manager.update_project(
            None,
            project_name,
            mlrun.api.schemas.ProjectUpdate(name="different-name"),
        )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name
    )


def test_delete_project(
    db: sqlalchemy.orm.Session,
    projects_manager: mlrun.api.utils.projects.manager.ProjectsManager,
    nop_consumer: mlrun.api.utils.projects.consumers.base.Consumer,
    nop_master: mlrun.api.utils.projects.consumers.base.Consumer,
):
    project_name = "project-name"
    projects_manager.create_project(
        None, mlrun.api.schemas.ProjectCreate(name=project_name),
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
        None, mlrun.api.schemas.ProjectCreate(name=project_name),
    )
    _assert_project_in_consumers([nop_master, nop_consumer], project_name)

    # add some project to consumer
    nop_consumer.create_project(
        None, mlrun.api.schemas.ProjectCreate(name="some-other-project")
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
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )
    _assert_project_in_consumers(
        [nop_master, nop_consumer], project_name, project_description
    )

    # change project description in consumer
    nop_consumer.update_project(
        None,
        project_name,
        mlrun.api.schemas.ProjectUpdate(description="updated description"),
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
