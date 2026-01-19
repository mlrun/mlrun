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
