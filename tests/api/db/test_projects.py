import pytest
import deepdiff
import typing
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import (
    _classes,
    Function,
    Project,
    Run,
    Artifact,
)
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_delete_project(db: DBInterface, db_session: Session):
    project_to_keep = "project_to_keep"
    project_to_remove = "project_to_remove"
    _create_resources_of_all_kinds(db, db_session, project_to_keep)
    _create_resources_of_all_kinds(db, db_session, project_to_remove)
    project_to_keep_table_name_records_count_map_before_project_removal = _assert_resources_in_project(
        db, db_session, project_to_keep
    )
    _assert_resources_in_project(db, db_session, project_to_remove)
    db.delete_project(db_session, project_to_remove)
    project_to_keep_table_name_records_count_map_after_project_removal = _assert_resources_in_project(
        db, db_session, project_to_keep
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


def _assert_resources_in_project(
    db: DBInterface,
    db_session: Session,
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
            if cls.__name__ != "Label" and cls.__name__ != "Project":
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


def _create_resources_of_all_kinds(db: DBInterface, db_session: Session, project: str):
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
        )
