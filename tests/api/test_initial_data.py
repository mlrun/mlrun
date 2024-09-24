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
#
import unittest.mock

import pytest
import sqlalchemy.exc
import sqlalchemy.orm

import mlrun
import mlrun.common.db.sql_session
import mlrun.common.schemas
import server.api.db.init_db
import server.api.db.sqldb.db
import server.api.initial_data
import server.api.utils.singletons.db
from mlrun.config import config


def test_add_data_version_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = server.api.initial_data.latest_data_version
    server.api.initial_data.latest_data_version = "3"
    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    server.api.initial_data._add_initial_data(db_session)
    assert (
        db.get_current_data_version(db_session, raise_on_not_found=True)
        == server.api.initial_data.latest_data_version
    )
    server.api.initial_data.latest_data_version = original_latest_data_version


def test_add_data_version_non_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = server.api.initial_data.latest_data_version
    server.api.initial_data.latest_data_version = "3"

    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    # fill db
    db.create_project(
        db_session,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    server.api.initial_data._add_initial_data(db_session)
    assert db.get_current_data_version(db_session, raise_on_not_found=True) == "1"
    server.api.initial_data.latest_data_version = original_latest_data_version


def test_perform_data_migrations_from_first_version():
    db, db_session = _initialize_db_without_migrations()

    # set version to 1
    db.create_data_version(db_session, "1")

    # keep a reference to the original functions, so we can restore them later
    original_perform_version_2_data_migrations = (
        server.api.initial_data._perform_version_2_data_migrations
    )
    server.api.initial_data._perform_version_2_data_migrations = unittest.mock.Mock()
    original_perform_version_3_data_migrations = (
        server.api.initial_data._perform_version_3_data_migrations
    )
    server.api.initial_data._perform_version_3_data_migrations = unittest.mock.Mock()
    original_perform_version_4_data_migrations = (
        server.api.initial_data._perform_version_4_data_migrations
    )
    server.api.initial_data._perform_version_4_data_migrations = unittest.mock.Mock()
    original_perform_version_5_data_migrations = (
        server.api.initial_data._perform_version_5_data_migrations
    )
    server.api.initial_data._perform_version_5_data_migrations = unittest.mock.Mock()
    original_perform_version_6_data_migrations = (
        server.api.initial_data._perform_version_6_data_migrations
    )
    server.api.initial_data._perform_version_6_data_migrations = unittest.mock.Mock()
    original_perform_version_7_data_migrations = (
        server.api.initial_data._perform_version_7_data_migrations
    )
    server.api.initial_data._perform_version_7_data_migrations = unittest.mock.Mock()

    original_perform_version_8_data_migrations = (
        server.api.initial_data._perform_version_8_data_migrations
    )
    server.api.initial_data._perform_version_8_data_migrations = unittest.mock.Mock()

    # perform migrations
    server.api.initial_data._perform_data_migrations(db_session)

    # calling again should not trigger migrations again, since we're already at the latest version
    server.api.initial_data._perform_data_migrations(db_session)

    server.api.initial_data._perform_version_2_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_3_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_4_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_5_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_6_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_7_data_migrations.assert_called_once()
    server.api.initial_data._perform_version_8_data_migrations.assert_called_once()

    assert db.get_current_data_version(db_session, raise_on_not_found=True) == str(
        server.api.initial_data.latest_data_version
    )

    # restore original functions
    server.api.initial_data._perform_version_2_data_migrations = (
        original_perform_version_2_data_migrations
    )
    server.api.initial_data._perform_version_3_data_migrations = (
        original_perform_version_3_data_migrations
    )
    server.api.initial_data._perform_version_4_data_migrations = (
        original_perform_version_4_data_migrations
    )
    server.api.initial_data._perform_version_5_data_migrations = (
        original_perform_version_5_data_migrations
    )
    server.api.initial_data._perform_version_6_data_migrations = (
        original_perform_version_6_data_migrations
    )
    server.api.initial_data._perform_version_7_data_migrations = (
        original_perform_version_7_data_migrations
    )
    server.api.initial_data._perform_version_8_data_migrations = (
        original_perform_version_8_data_migrations
    )


def test_resolve_current_data_version_version_exists():
    db, db_session = _initialize_db_without_migrations()

    db.create_data_version(db_session, "1")
    assert server.api.initial_data._resolve_current_data_version(db, db_session) == 1


@pytest.mark.parametrize("table_exists", [True, False])
@pytest.mark.parametrize("db_type", ["mysql", "sqlite"])
def test_resolve_current_data_version_before_and_after_projects(table_exists, db_type):
    db, db_session = _initialize_db_without_migrations()

    original_latest_data_version = server.api.initial_data.latest_data_version
    server.api.initial_data.latest_data_version = 3

    if not table_exists:
        # simulating table doesn't exist in DB
        db.get_current_data_version = unittest.mock.Mock()
        if db_type == "sqlite":
            db.get_current_data_version.side_effect = sqlalchemy.exc.OperationalError(
                "no such table", None, None
            )
        elif db_type == "mysql":
            db.get_current_data_version.side_effect = sqlalchemy.exc.ProgrammingError(
                "Table 'mlrun.data_versions' doesn't exist", None, None
            )

    assert server.api.initial_data._resolve_current_data_version(db, db_session) == 3
    # fill db
    db.create_project(
        db_session,
        mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    assert server.api.initial_data._resolve_current_data_version(db, db_session) == 1
    server.api.initial_data.latest_data_version = original_latest_data_version


def test_add_default_hub_source_if_needed():
    db, db_session = _initialize_db_without_migrations()

    # Start with no hub source
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
        raise_on_not_found=False,
    )
    assert hub_source is None

    # Create the default hub source
    server.api.initial_data._add_default_hub_source_if_needed(db, db_session)
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
    )
    assert hub_source.source.spec.path == config.hub.default_source.url

    # Change the config and make sure the hub source is updated
    config.hub.default_source.url = "http://some-other-url"
    server.api.initial_data._add_default_hub_source_if_needed(db, db_session)
    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
    )
    assert hub_source.source.spec.path == config.hub.default_source.url

    # Make sure the hub source is not updated if it already exists
    with unittest.mock.patch(
        "server.api.initial_data._update_default_hub_source"
    ) as update_default_hub_source:
        server.api.initial_data._add_default_hub_source_if_needed(db, db_session)
        assert update_default_hub_source.call_count == 0


def test_create_project_summaries():
    db, db_session = _initialize_db_without_migrations()

    # Create a project
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
    )

    with unittest.mock.patch.object(db, "_append_project_summary"):
        db.create_project(db_session, project)

    # Migrate the project summaries
    server.api.initial_data._create_project_summaries(db, db_session)

    # Check that the project summary was migrated
    migrated_project_summary = db.get_project_summary(db_session, project.metadata.name)

    assert migrated_project_summary.name == project.metadata.name


@pytest.mark.parametrize(
    "scheduled_object_labels, schedule_labels, expected_labels",
    [
        (
            {"label1": "value1"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value2"},
        ),
        ({"label1": "value1"}, {}, {"label1": "value1"}),
        ({}, {"label2": "value2"}, {"label2": "value2"}),
        (
            {"label1": "value1", "label3": "value3"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value2", "label3": "value3"},
        ),
        (
            {"label1": "value1", "label2": "value3"},
            {"label2": "value2"},
            {"label1": "value1", "label2": "value3"},
        ),
        (None, {"label2": "value2"}, {"label2": "value2"}),
        ({"label1": "value1"}, None, {"label1": "value1"}),
        (None, None, None),
    ],
)
def test_align_schedule_labels(
    scheduled_object_labels, schedule_labels, expected_labels
):
    db, db_session = _initialize_db_without_migrations()

    # Create a schedule
    db.create_schedule(
        session=db_session,
        project="project-name",
        name="schedule-name",
        kind=mlrun.common.schemas.ScheduleKinds.job,
        cron_trigger=mlrun.common.schemas.ScheduleCronTrigger.from_crontab("* * * * 1"),
        concurrency_limit=1,
        scheduled_object={"task": {"metadata": {"labels": scheduled_object_labels}}},
        labels=schedule_labels,
    )

    # Align schedule.labels and schedule.scheduled_object.task.metadata.labels
    db.align_schedule_labels(db_session)

    # Get updated schedules
    migrated_schedules = db.list_schedules(db_session)

    # Convert list[LabelRecord] to dict
    migrated_schedules_dict = {
        label.name: label.value for label in migrated_schedules[0].labels
    }

    assert (
        migrated_schedules[0].scheduled_object["task"]["metadata"]["labels"]
        or {} == migrated_schedules_dict
        or {} == expected_labels
    )


def _initialize_db_without_migrations() -> (
    tuple[server.api.db.sqldb.db.SQLDB, sqlalchemy.orm.Session]
):
    dsn = "sqlite:///:memory:?check_same_thread=false"
    mlrun.mlconf.httpdb.dsn = dsn
    mlrun.common.db.sql_session._init_engine(dsn=dsn)
    server.api.utils.singletons.db.initialize_db()
    db_session = mlrun.common.db.sql_session.create_session(dsn=dsn)
    db = server.api.db.sqldb.db.SQLDB(dsn)
    db.initialize(db_session)
    server.api.db.init_db.init_db()
    return db, db_session
