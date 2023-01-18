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
import typing
import unittest.mock

import pytest
import sqlalchemy.exc
import sqlalchemy.orm

import mlrun
import mlrun.api.db.init_db
import mlrun.api.db.sqldb.db
import mlrun.api.db.sqldb.session
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.singletons.db


def test_add_data_version_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = mlrun.api.initial_data.latest_data_version
    mlrun.api.initial_data.latest_data_version = "3"
    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    mlrun.api.initial_data._add_initial_data(db_session)
    assert (
        db.get_current_data_version(db_session, raise_on_not_found=True)
        == mlrun.api.initial_data.latest_data_version
    )
    mlrun.api.initial_data.latest_data_version = original_latest_data_version


def test_add_data_version_non_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = mlrun.api.initial_data.latest_data_version
    mlrun.api.initial_data.latest_data_version = "3"

    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    # fill db
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    mlrun.api.initial_data._add_initial_data(db_session)
    assert db.get_current_data_version(db_session, raise_on_not_found=True) == "1"
    mlrun.api.initial_data.latest_data_version = original_latest_data_version


def test_perform_data_migrations_from_zero_version():
    db, db_session = _initialize_db_without_migrations()

    # set version to 0
    db.create_data_version(db_session, "0")

    original_perform_version_1_data_migrations = (
        mlrun.api.initial_data._perform_version_1_data_migrations
    )
    mlrun.api.initial_data._perform_version_1_data_migrations = unittest.mock.Mock()

    mlrun.api.initial_data._perform_data_migrations(db_session)

    mlrun.api.initial_data._perform_version_1_data_migrations.assert_called_once()

    # calling again should trigger migrations again
    mlrun.api.initial_data._perform_data_migrations(db_session)

    mlrun.api.initial_data._perform_version_1_data_migrations.assert_called_once()

    mlrun.api.initial_data._perform_version_1_data_migrations = (
        original_perform_version_1_data_migrations
    )
    assert db.get_current_data_version(db_session, raise_on_not_found=True) == str(
        mlrun.api.initial_data.latest_data_version
    )


def test_resolve_current_data_version_version_exists():
    db, db_session = _initialize_db_without_migrations()

    db.create_data_version(db_session, "1")
    assert mlrun.api.initial_data._resolve_current_data_version(db, db_session) == 1


@pytest.mark.parametrize("table_exists", [True, False])
@pytest.mark.parametrize("db_type", ["mysql", "sqlite"])
def test_resolve_current_data_version_before_and_after_projects(table_exists, db_type):
    db, db_session = _initialize_db_without_migrations()

    original_latest_data_version = mlrun.api.initial_data.latest_data_version
    mlrun.api.initial_data.latest_data_version = 3

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

    assert mlrun.api.initial_data._resolve_current_data_version(db, db_session) == 3
    # fill db
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    assert mlrun.api.initial_data._resolve_current_data_version(db, db_session) == 1
    mlrun.api.initial_data.latest_data_version = original_latest_data_version


def _initialize_db_without_migrations() -> typing.Tuple[
    mlrun.api.db.sqldb.db.SQLDB, sqlalchemy.orm.Session
]:
    dsn = "sqlite:///:memory:?check_same_thread=false"
    mlrun.mlconf.httpdb.dsn = dsn
    mlrun.api.db.sqldb.session._init_engine(dsn=dsn)
    mlrun.api.utils.singletons.db.initialize_db()
    db_session = mlrun.api.db.sqldb.session.create_session()
    db = mlrun.api.db.sqldb.db.SQLDB(dsn)
    db.initialize(db_session)
    mlrun.api.db.init_db.init_db(db_session)
    return db, db_session
