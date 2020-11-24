import datetime
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
        mlrun.api.schemas.Project(name=project_name, description=project_description),
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
            mlrun.api.schemas.Project(
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
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )

    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.name == project_name
    assert project_output.description == project_description


def test_create_and_store_project_with_created(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_created = datetime.datetime.utcnow()

    mlrun_consumer.create_project(
        db,
        mlrun.api.schemas.Project(name=project_name, created=project_created),
    )

    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.name == project_name

    # Created in request body should be ignored and set by the DB layer
    assert project_output.created != project_created

    project_name_2 = "project-name-2"
    # first time - store will create
    mlrun_consumer.store_project(
        db,
        project_name_2,
        mlrun.api.schemas.Project(name=project_name_2, created=project_created),
    )

    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name_2
    )
    assert project_output.name == project_name_2

    # Created in request body should be ignored and set by the DB layer
    assert project_output.created != project_created

    # another time - this time store will update
    mlrun_consumer.store_project(
        db,
        project_name_2,
        mlrun.api.schemas.Project(name=project_name_2, created=project_created),
    )

    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name_2
    )
    assert project_output.name == project_name_2

    # Created in request body should be ignored and set by the DB layer
    assert project_output.created != project_created



def test_store_project_creation(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun_consumer.store_project(
        db,
        project_name,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.name == project_name
    assert project_output.description == project_description


def test_store_project_update(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )

    mlrun_consumer.store_project(
        db, project_name, mlrun.api.schemas.Project(name=project_name),
    )
    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.name == project_name
    assert project_output.description is None


def test_patch_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )

    updated_project_description = "some description 2"
    mlrun_consumer.patch_project(
        db,
        project_name,
        mlrun.api.schemas.ProjectPatch(description=updated_project_description),
    )
    project_output = mlrun.api.utils.singletons.db.get_db().get_project(
        db, project_name
    )
    assert project_output.name == project_name
    assert project_output.description == updated_project_description


def test_delete_project(
    db: sqlalchemy.orm.Session,
    mlrun_consumer: mlrun.api.utils.projects.consumers.mlrun.Consumer,
):
    project_name = "project-name"
    project_description = "some description"
    mlrun.api.utils.singletons.db.get_db().create_project(
        db,
        mlrun.api.schemas.Project(name=project_name, description=project_description),
    )
    mlrun_consumer.delete_project(db, project_name)

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlrun.api.utils.singletons.db.get_db().get_project(db, project_name)
