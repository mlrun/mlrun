import datetime
import typing
import unittest.mock

import deepdiff
import pytest
import sqlalchemy.orm

import mlrun.api.schemas
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import (
    _classes,
    Function,
    Project,
    Run,
    Artifact,
    FeatureSet,
    Feature,
    Entity,
    Schedule,
    FeatureVector,
)
import mlrun.api.initial_data
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_delete_project_with_resources(
    db: DBInterface, db_session: sqlalchemy.orm.Session
):
    project_to_keep = "project-to-keep"
    project_to_remove = "project-to-remove"
    _create_resources_of_all_kinds(db, db_session, project_to_keep)
    _create_resources_of_all_kinds(db, db_session, project_to_remove)
    project_to_keep_table_name_records_count_map_before_project_removal = (
        _assert_resources_in_project(db, db_session, project_to_keep)
    )
    _assert_resources_in_project(db, db_session, project_to_remove)
    # deletion strategy - restrict - should fail because there are resources
    with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
        db.delete_project(
            db_session, project_to_remove, mlrun.api.schemas.DeletionStrategy.restrict
        )
    # deletion strategy - cascade - should succeed and remove all related resources
    db.delete_project(
        db_session, project_to_remove, mlrun.api.schemas.DeletionStrategy.cascade
    )

    project_to_keep_table_name_records_count_map_after_project_removal = (
        _assert_resources_in_project(db, db_session, project_to_keep)
    )
    _assert_resources_in_project(
        db, db_session, project_to_remove, assert_no_resources=True
    )
    assert (
        deepdiff.DeepDiff(
            project_to_keep_table_name_records_count_map_before_project_removal,
            project_to_keep_table_name_records_count_map_after_project_removal,
            ignore_order=True,
        )
        == {}
    )

    # deletion strategy - restrict - should succeed cause no project
    db.delete_project(
        db_session, project_to_remove, mlrun.api.schemas.DeletionStrategy.restrict
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_data_migration_fill_project_state(
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
    mlrun.api.initial_data._fill_project_state(db, db_session)
    projects = db.list_projects(db_session)
    for project in projects.projects:
        assert project.spec.desired_state == mlrun.api.schemas.ProjectState.online
        assert project.status.state == project.spec.desired_state
    # verify not storing for no reason
    db.store_project = unittest.mock.Mock()
    mlrun.api.initial_data._fill_project_state(db, db_session)
    assert db.store_project.call_count == 0


def _generate_and_insert_pre_060_record(
    db_session: sqlalchemy.orm.Session, project_name: str
):
    pre_060_record = Project(name=project_name)
    db_session.add(pre_060_record)
    db_session.commit()


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_create_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_created = datetime.datetime.utcnow()

    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                created=project_created,
                labels=project_labels,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )

    project_output = db.get_project(db_session, project_name)
    assert project_output.metadata.name == project_name
    assert project_output.spec.description == project_description
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project_created
    assert (
        deepdiff.DeepDiff(
            project_labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_project_creation(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    project_created = datetime.datetime.utcnow()
    project_labels = {
        "some-label": "some-label-value",
    }
    db.store_project(
        db_session,
        project_name,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                created=project_created,
                labels=project_labels,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )
    project_output = db.get_project(db_session, project_name)
    assert project_output.metadata.name == project_name
    assert project_output.spec.description == project_description
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project_created
    assert (
        deepdiff.DeepDiff(
            project_labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_project_update(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    project_labels = {
        "some-label": "some-label-value",
    }
    project_created = datetime.datetime.utcnow()
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(
                name=project_name,
                created=project_created,
                labels=project_labels,
            ),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )

    db.store_project(
        db_session,
        project_name,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
        ),
    )
    project_output = db.get_project(db_session, project_name)
    assert project_output.metadata.name == project_name
    assert project_output.spec.description is None
    assert project_output.metadata.labels is None
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project_created


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_patch_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
):
    project_name = "project-name"
    project_description = "some description"
    project_created = datetime.datetime.utcnow()
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name=project_name),
            spec=mlrun.api.schemas.ProjectSpec(description=project_description),
        ),
    )

    patched_project_description = "some description 2"
    patched_project_labels = {
        "some-label": "some-label-value",
    }
    db.patch_project(
        db_session,
        project_name,
        {
            "metadata": {"created": project_created, "labels": patched_project_labels},
            "spec": {"description": patched_project_description},
        },
    )
    project_output = db.get_project(db_session, project_name)
    assert project_output.metadata.name == project_name
    assert project_output.spec.description == patched_project_description
    # Created in request body should be ignored and set by the DB layer
    assert project_output.metadata.created != project_created
    assert (
        deepdiff.DeepDiff(
            patched_project_labels,
            project_output.metadata.labels,
            ignore_order=True,
        )
        == {}
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
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


def _assert_resources_in_project(
    db: DBInterface,
    db_session: sqlalchemy.orm.Session,
    project: str,
    assert_no_resources: bool = False,
) -> typing.Dict:
    table_name_records_count_map = {}
    for cls in _classes:
        # User support is not really implemented or in use
        # Run tags support is not really implemented or in use
        if cls.__name__ != "User" and cls.__tablename__ != "runs_tags":
            number_of_cls_records = 0
            # Label doesn't have project attribute
            # Project (obviously) doesn't have project attribute
            # Features and Entities are not directly linked to project since they are sub-entity of feature-sets
            if (
                cls.__name__ != "Label"
                and cls.__name__ != "Project"
                and cls.__name__ != "Feature"
                and cls.__name__ != "Entity"
            ):
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
            else:
                number_of_cls_records = (
                    db_session.query(Project).filter(Project.name == project).count()
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


def _create_resources_of_all_kinds(
    db: DBInterface, db_session: sqlalchemy.orm.Session, project: str
):
    # add labels to project
    project_schema = schemas.Project(
        metadata=schemas.ProjectMetadata(name=project, labels={"key": "value"}),
        spec=schemas.ProjectSpec(description="some desc"),
    )
    db.store_project(db_session, project, project_schema)

    # Create several functions with several tags
    labels = {
        "name": "value",
        "name2": "value2",
    }
    function = {
        "bla": "blabla",
        "metadata": {"labels": labels},
        "status": {"bla": "blabla"},
    }
    function_names = ["function_name_1", "function_name_2", "function_name_3"]
    function_tags = ["some_tag", "some_tag2", "some_tag3"]
    for function_name in function_names:
        for function_tag in function_tags:
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
    artifact_tags = ["some_tag", "some_tag2", "some_tag3"]
    for artifact_key in artifact_keys:
        for artifact_uid in artifact_uids:
            for artifact_tag in artifact_tags:
                for artifact_iter in range(3):
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
        "metadata": {"labels": labels},
        "status": {"bla": "blabla"},
    }
    run_uids = ["some_uid", "some_uid2", "some_uid3"]
    for run_uid in run_uids:
        for run_iter in range(3):
            db.store_run(db_session, run, run_uid, project, run_iter)

    # Create several logs
    log = b"some random log"
    log_uids = ["some_uid", "some_uid2", "some_uid3"]
    for log_uid in log_uids:
        db.store_log(db_session, log_uid, project, log)

    # Create several schedule
    schedule = {
        "bla": "blabla",
        "status": {"bla": "blabla"},
    }
    schedule_cron_trigger = schemas.ScheduleCronTrigger(year=1999)
    schedule_names = ["schedule_name_1", "schedule_name_2", "schedule_name_3"]
    for schedule_name in schedule_names:
        db.create_schedule(
            db_session,
            project,
            schedule_name,
            schemas.ScheduleKinds.job,
            schedule,
            schedule_cron_trigger,
            labels,
        )

    feature_set = schemas.FeatureSet(
        metadata=schemas.ObjectMetadata(
            name="dummy", tag="latest", labels={"owner": "nobody"}
        ),
        spec=schemas.FeatureSetSpec(
            entities=[
                schemas.Entity(name="ent1", value_type="str", labels={"label": "1"})
            ],
            features=[
                schemas.Feature(name="feat1", value_type="str", labels={"label": "1"})
            ],
        ),
        status={},
    )
    db.create_feature_set(db_session, project, feature_set)

    feature_vector = schemas.FeatureVector(
        metadata=schemas.ObjectMetadata(
            name="dummy", tag="latest", labels={"owner": "somebody"}
        ),
        spec=schemas.ObjectSpec(),
        status=schemas.ObjectStatus(state="created"),
    )
    db.create_feature_vector(db_session, project, feature_vector)
