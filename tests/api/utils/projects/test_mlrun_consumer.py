import pytest
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.projects.consumers.base
import mlrun.api.utils.projects.consumers.mlrun
import mlrun.api.utils.projects.manager
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.errors


@pytest.fixture()
async def mlrun_consumer() -> mlrun.api.utils.projects.consumers.mlrun.Consumer:
    mlrun_consumer = mlrun.api.utils.projects.consumers.mlrun.Consumer()
    return mlrun_consumer


def test_get_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )

    project_output = mlrun_consumer.get_project(db, project_name)
    assert project_output.name == project_name
    assert project_output.description == project_description


def test_list_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    expected_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3"},
        {"name": "project-name-4", "description": "project-description-4"},
    ]
    for project in expected_projects:
        mlrun.api.utils.singletons.db.get_db().create_project(
            db,
            mlrun.api.schemas.ProjectCreate(
                name=project["name"], description=project.get("description")
            ),
        )
    projects_output = mlrun_consumer.list_projects(db)
    for index, project in enumerate(projects_output.projects):
        assert project.name == expected_projects[index]["name"]
        assert project.description == expected_projects[index].get("description")


def test_create_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"

    mlrun_consumer.create_project(
        db,
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )

    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.project.name == project_name
    assert project_output.project.description == project_description


def test_update_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )

    updated_project_description = "some description 2"
    mlrun_consumer.update_project(
        db,
        project_name,
        mlrun.api.schemas.ProjectUpdate(description=updated_project_description),
    )
    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.project.name == project_name
    assert project_output.project.description == updated_project_description


def test_delete_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.ProjectCreate(
            name=project_name, description=project_description
        ),
    )
    mlrun_consumer.delete_project(db, project_name)

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlrun.api.utils.singletons.db.get_db().get_project(db, project_name)
