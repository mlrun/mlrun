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
            # other errors
            ("some other exception", mlrun.errors.MLRunRuntimeError),
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
