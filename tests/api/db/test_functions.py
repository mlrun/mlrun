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
import pytest
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.models import Function
from tests.api.db.conftest import dbs


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_default_to_latest(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    function_hash_key = db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name
    )
    assert function_hash_key is not None
    function_queried_without_tag = db.get_function(db_session, function_1.metadata.name)
    function_queried_without_tag_hash = function_queried_without_tag["metadata"]["hash"]
    assert function_hash_key == function_queried_without_tag_hash
    assert function_queried_without_tag["metadata"]["tag"] == "latest"
    function_queried_with_tag = db.get_function(
        db_session, function_1.metadata.name, tag="latest"
    )
    function_queried_without_tag_hash = function_queried_with_tag["metadata"]["hash"]
    assert function_queried_with_tag is not None
    assert function_queried_with_tag["metadata"]["tag"] == "latest"
    assert function_queried_without_tag_hash == function_queried_without_tag_hash


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_versioned(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    function_hash_key = db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name, versioned=True
    )
    function_queried_without_hash_key = db.get_function(
        db_session, function_1.metadata.name
    )
    assert function_queried_without_hash_key is not None
    assert function_queried_without_hash_key["metadata"]["tag"] == "latest"

    # Verifying versioned function is queryable by hash_key
    function_queried_with_hash_key = db.get_function(
        db_session, function_1.metadata.name, hash_key=function_hash_key
    )
    function_queried_with_hash_key_hash = function_queried_with_hash_key["metadata"][
        "hash"
    ]
    assert function_queried_with_hash_key is not None
    assert function_queried_with_hash_key["metadata"]["tag"] == ""
    assert function_queried_with_hash_key_hash == function_hash_key

    function_2 = {"test": "new_version"}
    db.store_function(db_session, function_2, function_1.metadata.name, versioned=True)
    functions = db.list_functions(db_session, function_1.metadata.name)

    # Verifying both versions of the functions were saved
    assert len(functions) == 2

    tagged_count = 0
    for function in functions:
        if function["metadata"]["tag"] == "latest":
            tagged_count += 1

    # but only one was tagged
    assert tagged_count == 1


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_not_versioned(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    function_hash_key = db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name, versioned=False
    )
    function_result_1 = db.get_function(db_session, function_1.metadata.name)
    assert function_result_1 is not None
    assert function_result_1["metadata"]["tag"] == "latest"

    # not versioned so not queryable by hash key
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_function(
            db_session, function_1.metadata.name, hash_key=function_hash_key
        )

    function_2 = {"test": "new_version"}
    db.store_function(db_session, function_2, function_1.metadata.name, versioned=False)
    functions = db.list_functions(db_session, function_1.metadata.name)

    # Verifying only the latest version was saved
    assert len(functions) == 1


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_get_function_by_hash_key(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    function_hash_key = db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name, versioned=True
    )
    function_queried_without_hash_key = db.get_function(
        db_session, function_1.metadata.name
    )
    assert function_queried_without_hash_key is not None

    # Verifying function is queryable by hash_key
    function_queried_with_hash_key = db.get_function(
        db_session, function_1.metadata.name, hash_key=function_hash_key
    )
    assert function_queried_with_hash_key is not None

    # function queried by hash shouldn't have tag
    assert function_queried_without_hash_key["metadata"]["tag"] == "latest"
    assert function_queried_with_hash_key["metadata"]["tag"] == ""


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_get_function_by_tag(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    function_hash_key = db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name, versioned=True
    )
    function_queried_by_hash_key = db.get_function(
        db_session, function_1.metadata.name, hash_key=function_hash_key
    )
    function_not_queried_by_tag_hash = function_queried_by_hash_key["metadata"]["hash"]
    assert function_hash_key == function_not_queried_by_tag_hash


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_get_function_not_found(db: DBInterface, db_session: Session):
    function_1 = _generate_function()
    db.store_function(
        db_session, function_1.to_dict(), function_1.metadata.name, versioned=True
    )

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_function(db_session, function_1.metadata.name, tag="inexistent_tag")

    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_function(
            db_session, function_1.metadata.name, hash_key="inexistent_hash_key"
        )


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_list_functions_no_tags(db: DBInterface, db_session: Session):
    function_1 = {"bla": "blabla", "status": {"bla": "blabla"}}
    function_2 = {"bla2": "blabla", "status": {"bla": "blabla"}}
    function_name_1 = "function_name_1"

    # It is impossible to create a function without tag - only to create with a tag, and then tag another function with
    # the same tag
    tag = "some_tag"
    function_1_hash_key = db.store_function(
        db_session, function_1, function_name_1, tag=tag, versioned=True
    )
    function_2_hash_key = db.store_function(
        db_session, function_2, function_name_1, tag=tag, versioned=True
    )
    assert function_1_hash_key != function_2_hash_key
    functions = db.list_functions(db_session, function_name_1)
    assert len(functions) == 2

    # Verify function 1 without tag and has not status
    for function in functions:
        if function["metadata"]["hash"] == function_1_hash_key:
            assert function["metadata"]["tag"] == ""
            assert function["status"] is None


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_list_functions_by_tag(db: DBInterface, db_session: Session):
    tag = "function_name_1"

    names = ["some_name", "some_name2", "some_name3"]
    for name in names:
        function_body = {"metadata": {"name": name}}
        db.store_function(db_session, function_body, name, tag=tag, versioned=True)
    functions = db.list_functions(db_session, tag=tag)
    assert len(functions) == len(names)
    for function in functions:
        function_name = function["metadata"]["name"]
        names.remove(function_name)
    assert len(names) == 0


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_functions_with_non_existent_tag(db: DBInterface, db_session: Session):
    names = ["some_name", "some_name2", "some_name3"]
    for name in names:
        function_body = {"metadata": {"name": name}}
        db.store_function(db_session, function_body, name, versioned=True)
    functions = db.list_functions(db_session, tag="non_existent_tag")
    assert len(functions) == 0


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_list_functions_filtering_unversioned_untagged(
    db: DBInterface, db_session: Session
):
    function_1 = _generate_function()
    function_2 = _generate_function()
    tag = "some_tag"
    db.store_function(
        db_session,
        function_1.to_dict(),
        function_1.metadata.name,
        versioned=False,
        tag=tag,
    )
    tagged_function_hash_key = db.store_function(
        db_session,
        function_2.to_dict(),
        function_2.metadata.name,
        versioned=True,
        tag=tag,
    )
    functions = db.list_functions(db_session, function_1.metadata.name)

    # First we stored to the tag without versioning (unversioned instance) then we stored to the tag with version
    # so the unversioned instance remained untagged, verifying we're not getting it
    assert len(functions) == 1
    assert functions[0]["metadata"]["hash"] == tagged_function_hash_key


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_delete_function(db: DBInterface, db_session: Session):
    labels = {
        "name": "value",
        "name2": "value2",
    }
    function = {
        "bla": "blabla",
        "metadata": {"labels": labels},
        "status": {"bla": "blabla"},
    }
    function_name = "function_name_1"
    project = "bla"
    tags = ["some_tag", "some_tag2", "some_tag3"]
    function_hash_key = None
    for tag in tags:
        function_hash_key = db.store_function(
            db_session, function, function_name, project, tag=tag, versioned=True
        )

    # if not exploding then function exists
    for tag in tags:
        db.get_function(db_session, function_name, project, tag=tag)
    db.get_function(db_session, function_name, project, hash_key=function_hash_key)
    assert len(tags) == len(db.list_functions(db_session, function_name, project))
    number_of_tags = (
        db_session.query(Function.Tag)
        .filter_by(project=project, obj_name=function_name)
        .count()
    )
    number_of_labels = db_session.query(Function.Label).count()

    assert len(tags) == number_of_tags
    assert len(labels) == number_of_labels

    db.delete_function(db_session, project, function_name)

    for tag in tags:
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            db.get_function(db_session, function_name, project, tag=tag)
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.get_function(db_session, function_name, project, hash_key=function_hash_key)
    assert 0 == len(db.list_functions(db_session, function_name, project))

    # verifying tags and labels (different table) records were removed
    number_of_tags = (
        db_session.query(Function.Tag)
        .filter_by(project=project, obj_name=function_name)
        .count()
    )
    number_of_labels = db_session.query(Function.Label).count()

    assert number_of_tags == 0
    assert number_of_labels == 0


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
@pytest.mark.parametrize("use_hash_key", [True, False])
def test_list_functions_multiple_tags(
    db: DBInterface, db_session: Session, use_hash_key: bool
):
    function_1 = _generate_function()

    tags = ["some_tag", "some_tag2", "some_tag3"]
    for tag in tags:
        function_hash_key = db.store_function(
            db_session,
            function_1.to_dict(),
            function_1.metadata.name,
            tag=tag,
            versioned=True,
        )
    functions = db.list_functions(
        db_session,
        function_1.metadata.name,
        hash_key=function_hash_key if use_hash_key else None,
    )
    assert len(functions) == len(tags)
    for function in functions:
        if use_hash_key:
            assert function["metadata"]["hash"] == function_hash_key
        function_tag = function["metadata"]["tag"]
        tags.remove(function_tag)
    assert len(tags) == 0


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_function_with_tag_and_uid(db: DBInterface, db_session: Session):
    tag_name = "some_tag"
    function_1 = _generate_function(tag=tag_name)
    function_2 = _generate_function(function_name="function_name_2", tag=tag_name)

    function_1_hash_key = db.store_function(
        db_session,
        function_1.to_dict(),
        function_1.metadata.name,
        tag=tag_name,
        versioned=True,
    )

    # Storing another function with the same tag,
    # to ensure that filtering by tag and hash key works, and that not both are returned
    db.store_function(
        db_session,
        function_2.to_dict(),
        function_2.metadata.name,
        tag=tag_name,
        versioned=True,
    )

    functions = db.list_functions(
        db_session, function_1.metadata.name, tag=tag_name, hash_key=function_1_hash_key
    )
    assert (
        len(functions) == 1 and functions[0]["metadata"]["hash"] == function_1_hash_key
    )


def _generate_function(
    function_name: str = "function_name_1",
    project: str = "project_name",
    tag: str = "latest",
):
    return mlrun.new_function(
        name=function_name,
        project=project,
        tag=tag,
    )
