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

import alembic.config
import pytest
import pytest_alembic.plugin.fixtures
from pytest_alembic.tests import (  # noqa
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
)

import framework.db.sqldb.models


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
def alembic_runner(alembic_engine):
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
def before_revision_data():
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
def notifications_test_alembic_runner(alembic_engine, before_revision_data):
    config = pytest_alembic.plugin.fixtures.Config(
        alembic_config=alembic.config.Config(
            file_=Constants.ini_file_path,
        ),
        before_revision_data=before_revision_data,
    )
    with pytest_alembic.runner(
        config=config,
        engine=alembic_engine,
    ) as runner:
        yield runner


@pytest.mark.alembic
def test_notification_params_to_secret_params(
    notifications_test_alembic_runner,
    alembic_session,
    before_revision_data,
):
    notifications_test_alembic_runner.migrate_up_to(
        Constants.notifications_params_to_secret_params_revision
    )

    for index, item in enumerate(
        alembic_session.query(
            framework.db.sqldb.models.Run.Notification.params,
            framework.db.sqldb.models.Run.Notification.secret_params,
        )
        .filter_by(project=Constants.notifications_params_to_secret_params_project)
        .order_by(framework.db.sqldb.models.Run.Notification.id)
    ):
        assert not item.params
        assert (
            item.secret_params
            == before_revision_data[
                Constants.notifications_params_to_secret_params_revision
            ][index]["params"]
        )
