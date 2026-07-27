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
from unittest import mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

import mlrun.common.schemas
import mlrun.utils.helpers

import framework.db.sqldb.db
import framework.db.sqldb.models
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestSQLDB(TestDatabaseBase):
    @pytest.mark.parametrize(
        "error_message, expected_exception",
        [
            # exhausted retries
            ("database is locked", Exception),
            # conflicts
            (
                "(sqlite3.IntegrityError) UNIQUE constraint failed",
                mlrun.errors.MLRunConflictError,
            ),
            ("(pymysql.err.IntegrityError) (1062", mlrun.errors.MLRunConflictError),
            ("(pymysql.err.IntegrityError) (1586", mlrun.errors.MLRunConflictError),
            (
                "(psycopg2.errors.UniqueViolation) duplicate key value violates "
                'unique constraint "_functions_uc"',
                mlrun.errors.MLRunConflictError,
            ),
            (
                "(psycopg.errors.UniqueViolation) duplicate key value violates "
                'unique constraint "_functions_uc"',
                mlrun.errors.MLRunConflictError,
            ),
            # other errors
            ("some other exception", mlrun.errors.MLRunRuntimeError),
            # Postgres non-unique integrity violations must stay fatal, not be
            # misclassified as retryable conflicts.
            (
                "(psycopg2.errors.NotNullViolation) null value in column",
                mlrun.errors.MLRunRuntimeError,
            ),
            (
                "(psycopg2.errors.ForeignKeyViolation) insert or update on table",
                mlrun.errors.MLRunRuntimeError,
            ),
        ],
    )
    def test_commit_failures(self, error_message: str, expected_exception: Exception):
        # create some fake objects to commit
        objects = [
            framework.db.sqldb.models.Run(project="p1", uid="u1", name="run-1"),
            framework.db.sqldb.models.Feature(feature_set_id="fs-1", name="feat-1"),
            framework.db.sqldb.models.Function(project="p3", name="func-1"),
        ]

        session = mock.MagicMock()
        session.commit = mock.MagicMock(side_effect=SQLAlchemyError(error_message))

        with pytest.raises(expected_exception):
            self._db._commit(session, objects)

    @pytest.mark.parametrize(
        "error_message, is_conflict",
        [
            # Postgres duplicate-key races, via both installed drivers (psycopg2 and
            # psycopg v3). These are the get-then-insert conflicts retry_on_conflict
            # must recognize so the store re-runs instead of failing fatally.
            (
                "(psycopg2.errors.UniqueViolation) duplicate key value violates "
                'unique constraint "_functions_uc"',
                True,
            ),
            (
                "(psycopg.errors.UniqueViolation) duplicate key value violates "
                'unique constraint "_functions_uc"',
                True,
            ),
            # Non-unique Postgres integrity violations are genuine failures and must
            # not be swallowed by the conflict-retry mechanism.
            ("(psycopg2.errors.NotNullViolation) null value in column", False),
            (
                "(psycopg.errors.ForeignKeyViolation) insert or update on table",
                False,
            ),
        ],
    )
    def test_conflict_messages_match_postgres_unique_violation(
        self, error_message: str, is_conflict: bool
    ):
        # retry_on_conflict (and the current-tree store_function _flush path) classify
        # a raw SQLAlchemy error by substring-matching its chain against conflict_messages.
        exc = SQLAlchemyError(error_message)
        assert (
            mlrun.utils.helpers.are_strings_in_exception_chain_messages(
                exc, framework.db.sqldb.db.conflict_messages
            )
            is is_conflict
        )
