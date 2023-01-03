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
import datetime
import unittest.mock

import deepdiff
import pytest
import sqlalchemy.orm

import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.errors
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import Project


def test_get_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name, labels=project_labels
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )

    project_output = db.get_project(db_session, project_name)
    assert project_output.metadata.name == project_name
    assert project_output.spec.description == project_description
    assert (
        deepdiff.DeepDiff(
            project_labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )


def test_get_project_with_pre_060_record(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project_name"
    _generate_and_insert_pre_060_record(db_session, project_name)
    pre_060_record = (
        db_session.query(Project).filter(Project.name == project_name).one()
    )
    assert pre_060_record.full_object is None
    project = db.get_project(
        db_session,
        project_name,
    )
    assert project.metadata.name == project_name
    updated_record = (
        db_session.query(Project).filter(Project.name == project_name).one()
    )
    # when GET performed on a project of the old format - we're upgrading it to the new format - ensuring it happened
    assert updated_record.full_object is not None


def test_data_migration_enrich_project_state(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    for i in range(10):
        project_name = f"project-name-{i}"
        _generate_and_insert_pre_060_record(db_session, project_name)
    projects = db.list_projects(db_session)
    for project in projects.projects:
        # getting default value from the schema
        assert project.spec.desired_state == mlrun.api.schemas.ProjectState.online
        assert project.status.state is None
    mlrun.api.initial_data._enrich_project_state(db, db_session)
    projects = db.list_projects(db_session)
    for project in projects.projects:
        assert project.spec.desired_state == mlrun.api.schemas.ProjectState.online
        assert project.status.state == project.spec.desired_state
    # verify not storing for no reason
    db.store_project = unittest.mock.Mock()
    mlrun.api.initial_data._enrich_project_state(db, db_session)
    assert db.store_project.call_count == 0


def _generate_and_insert_pre_060_record(
    db_session: sqlalchemy.orm.Session, project_name: str
):
    pre_060_record = Project(name=project_name)
    db_session.add(pre_060_record)
    db_session.commit()


def test_list_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    expected_projects = [
        {"name": "project-name-1"},
        {"name": "project-name-2", "description": "project-description-2"},
        {"name": "project-name-3", "labels": {"key": "value"}},
        {
            "name": "project-name-4",
            "description": "project-description-4",
            "labels": {"key2": "value2"},
        },
    ]
    for project in expected_projects:
        db.create_project(
            db_session,
            mlrun.api.schemas.Project(
                metadata=mlrun.api.schemas.ProjectMetadata(
                    name=project["name"], labels=project.get("labels")
                ),
                spec=mlrun.api.schemas.ProjectSpec(
                    description=project.get("description")
                ),
            ),
        )
    projects_output = db.list_projects(db_session)
    for index, project in enumerate(projects_output.projects):
        assert project.metadata.name == expected_projects[index]["name"]
        assert project.spec.description == expected_projects[index].get("description")
        assert (
            deepdiff.DeepDiff(
                expected_projects[index].get("labels"),
                project.metadata.labels,
                ignore_order=True,
            )
            == {}
        )


def test_list_project_names_filter(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):

    project_names = ["project-1", "project-2", "project-3", "project-4", "project-5"]
    for project in project_names:
        db.create_project(
            db_session,
            mlrun.api.schemas.Project(
                metadata=mlrun.api.schemas.ProjectMetadata(name=project),
            ),
        )
    filter_names = [project_names[0], project_names[3], project_names[4]]
    projects_output = db.list_projects(
        db_session,
        format_=mlrun.api.schemas.ProjectsFormat.name_only,
        names=filter_names,
    )

    assert (
        deepdiff.DeepDiff(
            filter_names,
            projects_output.projects,
            ignore_order=True,
        )
        == {}
    )

    projects_output = db.list_projects(
        db_session,
        format_=mlrun.api.schemas.ProjectsFormat.name_only,
        names=[],
    )

    assert projects_output.projects == []


def test_create_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project = _generate_project()
    db.create_project(
        db_session,
        project.copy(deep=True),
    )
    _assert_project(db, db_session, project)


def test_store_project_creation(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project = _generate_project()
    db.store_project(
        db_session,
        project.metadata.name,
        project.copy(deep=True),
    )
    _assert_project(db, db_session, project)


def test_store_project_update(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project = _generate_project()
    db.create_project(
        db_session,
        project.copy(deep=True),
    )

    db.store_project(
        db_session,
        project.metadata.name,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project.metadata.name),
        ),
    )
    project_output = db.get_project(db_session, project.metadata.name)
    assert project_output.metadata.name == project.metadata.name
    assert project_output.spec.description is None
    assert project_output.metadata.labels is None
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project.metadata.created


def test_patch_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project = _generate_project()
    db.create_project(
        db_session,
        project.copy(deep=True),
    )

    patched_project_description = "some description 2"
    patched_project_labels = {
        "some-label": "some-label-value",
    }
    db.patch_project(
        db_session,
        project.metadata.name,
        {
            "metadata": {
                "created": project.metadata.created,
                "labels": patched_project_labels,
            },
            "spec": {"description": patched_project_description},
        },
    )
    project_output = db.get_project(db_session, project.metadata.name)
    assert project_output.metadata.name == project.metadata.name
    assert project_output.spec.description == patched_project_description
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project.metadata.created
    assert (
        deepdiff.DeepDiff(
            patched_project_labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )


def test_delete_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )
    db.delete_project(db_session, project_name)

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_project(db_session, project_name)


def _generate_project():
    return mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(
            name="project-name",
            created=datetime.datetime.utcnow() - datetime.timedelta(seconds=1),
            labels={
                "some-label": "some-label-value",
            },
        ),
        spec=mlrun.api.schemas.ProjectSpec(
            description="some description", owner="owner-name"
        ),
    )


def _assert_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
    expected_project: mlrun.api.schemas.Project,
):
    project_output = db.get_project(db_session, expected_project.metadata.name)
    assert project_output.metadata.name == expected_project.metadata.name
    assert project_output.spec.description == expected_project.spec.description
    assert project_output.spec.owner == expected_project.spec.owner
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != expected_project.metadata.created
    assert (
        deepdiff.DeepDiff(
            expected_project.metadata.labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )
