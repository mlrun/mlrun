# Copyright 2024 Iguazio
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

import pytest
import sqlalchemy.orm

import framework.db.session
import framework.utils.singletons.db

SOME_VALUE = 5
POPULATE_VALUE = 10  # the value to populate the counter with in the 'populate' tests (to simplify the parametrizing)


@pytest.mark.parametrize(
    "action,existing_counter,expected_counter",
    [
        ("populate", None, POPULATE_VALUE),
        ("populate", 0, POPULATE_VALUE),
        ("populate", SOME_VALUE, POPULATE_VALUE),
        ("populate", POPULATE_VALUE, POPULATE_VALUE),
        ("increment", None, 1),
        ("increment", 0, 1),
        ("increment", SOME_VALUE, SOME_VALUE + 1),
        ("decrement", None, 0),
        ("decrement", 0, 0),
        ("decrement", SOME_VALUE, SOME_VALUE - 1),
        ("get_or_create", None, 0),
        ("get_or_create", 0, 0),
        ("get_or_create", SOME_VALUE, SOME_VALUE),
    ],
)
def test_object_counters(
    db: sqlalchemy.orm.Session, action, existing_counter, expected_counter
):
    project = "project"
    object_kind = "object_kind"
    object_subkind = "object_subkind"

    mldb = framework.utils.singletons.db.get_db()

    # if counter already exists for the test, populate it with the existing value
    if existing_counter is not None:
        object_counter = mldb.populate_object_counter(
            db, project, object_kind, object_subkind, existing_counter
        )
        assert object_counter.counter == existing_counter

    if action == "populate":
        # 'populate' is the only action that accepts a parameter
        object_counter = mldb.populate_object_counter(
            db, project, object_kind, object_subkind, POPULATE_VALUE
        )
        assert object_counter.counter == expected_counter
    else:
        # the rest of the actions only require the db session
        object_counter = getattr(mldb, f"{action}_object_counter")(
            db, project, object_kind, object_subkind
        )
        assert object_counter.counter == expected_counter

    mldb.delete_object_counter(db, project, object_kind, object_subkind)
    object_counter = mldb.get_or_create_object_counter(
        db, project, object_kind, object_subkind
    )
    assert object_counter.counter == 0
