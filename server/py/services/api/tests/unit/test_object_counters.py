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
import framework.utils.object_counters

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
        ("counter", None, 0),
        ("counter", 0, 0),
        ("counter", SOME_VALUE, SOME_VALUE),
    ],
)
def test_object_counters(
    db: sqlalchemy.orm.Session, action, existing_counter, expected_counter
):
    project = "project"
    object_kind = "object_kind"
    object_subkind = "object_subkind"

    # initialize the object counter instance (no db communication yet)
    object_counter = framework.utils.object_counters.ObjectCounter(
        project, object_kind, object_subkind
    )

    # if counter already exists for the test, populate it with the existing value
    if existing_counter is not None:
        counter = object_counter.populate(db, existing_counter)
        assert counter == existing_counter

    if action == "populate":
        # 'populate' is the only action that accepts a parameter
        counter = object_counter.populate(db, POPULATE_VALUE)
        assert counter == expected_counter
    else:
        # the rest of the actions only require the db session
        counter = getattr(object_counter, action)(db)
        assert counter == expected_counter

    object_counter.delete(db)
    assert object_counter.counter(db) == 0
