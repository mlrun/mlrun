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

import json
import pathlib
import typing

import alembic.config
import pytest
import pytest_alembic.plugin.fixtures
import sqlalchemy.orm
from pytest_alembic import MigrationContext
from pytest_alembic.tests import (  # noqa
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
)

import tests.conftest

pytest_plugins = [
    "tests.common_fixtures",
    "tests.conftest",
]


class Constants:
    ini_file_path = str(
        pathlib.Path(__file__).absolute().parent.parent.parent.parent / "alembic.ini"
    )
    notifications_table = "runs_notifications"

    notifications_params_to_secret_params_revision = "eefc169f7633"
    notifications_params_to_secret_params_project = (
        "notifications_params_to_secret_params_project"
    )

    bg_task_label_dedup_revision = "6d1d53f60e90"
    bg_task_label_dedup_pre_revision = "0da0066c77f5"


@pytest.fixture
def alembic_runner(
    alembic_engine: sqlalchemy.engine.Engine,
) -> typing.Generator[MigrationContext, None, None]:
    config = pytest_alembic.plugin.fixtures.Config(
        alembic_config=alembic.config.Config(
            file_=Constants.ini_file_path,
        ),
    )
    with pytest_alembic.runner(
        config=config,
        engine=alembic_engine,
    ) as runner:
        yield runner


@pytest.fixture
def before_revision_data() -> dict[str, list[dict[str, str]]]:
    return {
        Constants.notifications_params_to_secret_params_revision: [
            {
                "__tablename__": Constants.notifications_table,
                "project": Constants.notifications_params_to_secret_params_project,
                "name": "notifications1",
                "kind": "console",
                "message": "test",
                "severity": "info",
                "when": "completed",
                "params": json.dumps({"obj": {"x": 99}}),
                "condition": "",
                "status": "",
            }
        ],
    }


@pytest.fixture
def notifications_test_alembic_runner(
    db_engine: sqlalchemy.engine.Engine,
    before_revision_data: dict[str, list[dict[str, str]]],
) -> typing.Generator[MigrationContext, None, None]:
    config = pytest_alembic.plugin.fixtures.Config(
        alembic_config=alembic.config.Config(
            file_=Constants.ini_file_path,
        ),
        before_revision_data=before_revision_data,
    )
    with pytest_alembic.runner(
        config=config,
        engine=db_engine,
    ) as runner:
        yield runner


@pytest.mark.alembic
def test_notification_params_to_secret_params(
    alembic_session: sqlalchemy.orm.Session,
    notifications_test_alembic_runner: MigrationContext,
    before_revision_data: dict[str, list[dict[str, str]]],
):
    notifications_test_alembic_runner.migrate_up_to(
        Constants.notifications_params_to_secret_params_revision
    )
    from framework.db.sqldb.models import Run

    for index, item in enumerate(
        alembic_session.query(
            Run.Notification.params,
            Run.Notification.secret_params,
        )
        .filter_by(
            project=Constants.notifications_params_to_secret_params_project,
        )
        .order_by(Run.Notification.id)
    ):
        assert not item.params
        assert (
            item.secret_params
            == before_revision_data[
                Constants.notifications_params_to_secret_params_revision
            ][index]["params"]
        )


@pytest.mark.alembic
def test_background_task_label_migration_handles_duplicates(
    alembic_engine: sqlalchemy.engine.Engine,
    alembic_session: sqlalchemy.orm.Session,
    alembic_runner: MigrationContext,
):
    """Ensure migration 6d1d53f60e90 deduplicates labels that collide under the new (project, name, value)
    unique constraint.

    The old constraint was (task_id, name), so two labels with the same (name, value) could exist under different
    tasks in the same project.
    The migration must backfill the project column and remove such duplicates before applying the new constraint.
    """
    tests.conftest._wipe_database(alembic_engine)
    alembic_runner.migrate_up_to(Constants.bg_task_label_dedup_pre_revision)

    # Two tasks in the same project, one in a different project
    alembic_session.execute(
        sqlalchemy.text(
            "INSERT INTO background_tasks (id, name, project, state) VALUES "
            "(1, 'task-1', 'my-proj', 'running'), "
            "(2, 'task-2', 'my-proj', 'running'), "
            "(3, 'task-3', 'other-proj', 'running')"
        )
    )

    # Labels 1 & 2: same (name, value), different task_ids in the SAME project will become duplicates after project
    # backfill
    # Label 3: same (name, value) but parent task is in a DIFFERENT project, should be preserved
    alembic_session.execute(
        sqlalchemy.text(
            "INSERT INTO background_task_labels (id, task_id, name, value) VALUES "
            "(1, 1, 'workflow', 'pipeline-run-abc'), "
            "(2, 2, 'workflow', 'pipeline-run-abc'), "
            "(3, 3, 'workflow', 'pipeline-run-abc')"
        )
    )
    alembic_session.commit()

    # Run our migration
    alembic_runner.migrate_up_to(Constants.bg_task_label_dedup_revision)

    rows = alembic_session.execute(
        sqlalchemy.text(
            "SELECT id, task_id, project, name, value "
            "FROM background_task_labels ORDER BY id"
        )
    ).fetchall()

    # One duplicate removed from my-proj, other-proj label preserved
    assert len(rows) == 2, f"Expected 2 rows after dedup, got {len(rows)}"

    my_proj_rows = [r for r in rows if r.project == "my-proj"]
    assert len(my_proj_rows) == 1
    assert my_proj_rows[0].id == 2, "Expected the latest duplicate (id=2) to be kept"

    other_proj_rows = [r for r in rows if r.project == "other-proj"]
    assert len(other_proj_rows) == 1
    assert other_proj_rows[0].id == 3
