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
import datetime
import time

import pytest
from sqlalchemy.orm import Session

import mlrun.errors
from server.api.db.base import DBInterface
from server.api.db.sqldb.models import Function


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


def test_get_function_when_using_not_normalize_name(
    db: DBInterface, db_session: Session
):
    # add a function with a non-normalized name to the database
    function_name = "function_name"
    project_name = "project"
    _generate_and_insert_function_record(db_session, function_name, project_name)

    # getting the function using the non-normalized name, and ensure that it works
    response = db.get_function(db_session, function_name, project_name)
    assert response["metadata"]["name"] == function_name


def _generate_and_insert_function_record(
    db_session: Session, function_name: str, project_name: str
):
    function = {
        "metadata": {"name": function_name, "project": project_name},
        "spec": {"asd": "test"},
    }
    fn = Function(
        name=function_name, project=project_name, struct=function, uid="1", id="1"
    )
    tag = Function.Tag(project=project_name, name="latest", obj_name=fn.name)
    tag.obj_id, tag.uid = fn.id, fn.uid
    db_session.add(fn)
    db_session.add(tag)
    db_session.commit()


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


def test_list_functions_with_non_existent_tag(db: DBInterface, db_session: Session):
    names = ["some_name", "some_name2", "some_name3"]
    for name in names:
        function_body = {"metadata": {"name": name}}
        db.store_function(db_session, function_body, name, versioned=True)
    functions = db.list_functions(db_session, tag="non_existent_tag")
    assert len(functions) == 0


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


def test_list_functions_with_format(db: DBInterface, db_session: Session):
    name = "function_name_1"
    tag = "some_tag"
    function_body = {
        "metadata": {"name": name},
        "kind": "remote",
        "status": {"state": "online"},
        "spec": {
            "description": "some_description",
            "command": "some_command",
            "image": "some_image",
            "default_handler": "some_handler",
            "default_class": "some_class",
            "graph": "some_graph",
            "preemption_mode": "some_preemption_mode",
            "node_selector": {"some_node_selector": "value"},
            "priority_class_name": "some_priority_class_name",
            "extra_field": "extra_field",
        },
    }
    db.store_function(db_session, function_body, name, tag=tag, versioned=True)
    functions = db.list_functions(db_session, tag=tag, format_="full")
    assert len(functions) == 1
    function = functions[0]
    assert function["spec"] == function_body["spec"]

    functions = db.list_functions(db_session, tag=tag, format_="minimal")
    assert len(functions) == 1
    function = functions[0]
    del function_body["spec"]["extra_field"]
    assert function["spec"] == function_body["spec"]


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


def test_list_functions_with_time_filters(db: DBInterface, db_session: Session):
    function_1 = _generate_function("function-name-1")
    function_2 = _generate_function("function-name-2")
    function_3 = _generate_function("function-name-3")

    for function in [function_1, function_2, function_3]:
        db.store_function(
            db_session, function.to_dict(), function.metadata.name, versioned=True
        )
        time.sleep(1)

    # Verifying that the time filters are working:
    # No Filters
    all_functions = db.list_functions(db_session)
    assert len(all_functions) == 3

    # extract the updated time of the functions
    function_times = [
        function["metadata"]["updated"]
        for function in sorted(all_functions, key=lambda x: x["metadata"]["updated"])
    ]

    # Since only
    functions = db.list_functions(db_session, since=function_times[1])
    assert len(functions) == 2

    # Until only
    functions = db.list_functions(db_session, until=function_times[1])
    assert len(functions) == 2

    # Since and Until
    functions = db.list_functions(
        db_session, since=function_times[0], until=function_times[0]
    )
    assert len(functions) == 1

    # Since and Until with no results
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    functions = db.list_functions(db_session, until=yesterday)
    assert len(functions) == 0
    functions = db.list_functions(db_session, since=now)
    assert len(functions) == 0


def test_list_untagged_functions(db: DBInterface, db_session: Session):
    # create 2 functions, one with tag and one without
    function_1_name = "function-name-1"
    function_2_name = "function-name-2"
    function_1 = _generate_function(function_1_name)
    function_2 = _generate_function(function_2_name)
    tag = "some-tag"

    # function 1 should get the tag
    tagged_function_hash = db.store_function(
        db_session,
        function_1.to_dict(),
        function_1_name,
        tag=tag,
        versioned=True,
    )

    # function 2 should get the "latest" tag
    func_2_dict = function_2.to_dict()
    function_2_hash_key = db.store_function(
        db_session,
        func_2_dict,
        function_2_name,
        versioned=True,
    )

    # list all functions
    functions = db.list_functions(db_session)
    assert len(functions) == 2

    # change something in the second function
    func_2_dict["spec"]["command"] = "new_command"

    # store the function again, the new instance should get the "latest" tag
    db.store_function(
        db_session,
        func_2_dict,
        function_2_name,
        versioned=True,
    )

    # list all functions
    functions = db.list_functions(db_session)

    # list only tagged functions
    tagged_function = db.list_functions(db_session, tag="*")

    assert len(functions) != len(tagged_function)

    all_hashes = [function["metadata"]["hash"] for function in functions]
    untagged_hashes = [function["metadata"]["hash"] for function in tagged_function]

    assert function_2_hash_key in all_hashes
    assert function_2_hash_key not in untagged_hashes

    # list function with specific tag
    tagged_function = db.list_functions(db_session, tag=tag)
    assert len(tagged_function) == 1
    assert tagged_function[0]["metadata"]["hash"] == tagged_function_hash


def test_delete_functions(db: DBInterface, db_session: Session):
    names = ["some_name", "some_name2", "some_name3"]
    labels = {
        "key": "value",
    }
    function = {
        "bla": "blabla",
        "metadata": {"labels": labels},
        "status": {"bla": "blabla"},
    }
    for name in names:
        db.store_function(
            db_session, function, name, project="project1", tag="latest", versioned=True
        )
        db.store_function(
            db_session,
            function,
            name,
            project="project1",
            tag="latest_2",
            versioned=True,
        )
        db.store_function(
            db_session, function, name, project="project2", tag="latest", versioned=True
        )
        db.store_function(
            db_session,
            function,
            name,
            project="project2",
            tag="latest_2",
            versioned=True,
        )
    functions = db.list_functions(db_session, project="project1")
    assert len(functions) == len(names) * 2
    functions = db.list_functions(db_session, project="project2")
    assert len(functions) == len(names) * 2

    assert db_session.query(Function.Label).count() != 0
    assert db_session.query(Function.Tag).count() != 0
    assert db_session.query(Function).count() != 0

    db.delete_functions(db_session, "*", names=names[:2])
    functions = db.list_functions(db_session, project="project1")
    assert len(functions) == 2
    functions = db.list_functions(db_session, project="project2")
    assert len(functions) == 2

    assert db_session.query(Function.Label).count() == 2
    assert db_session.query(Function.Tag).count() == 4
    assert db_session.query(Function).count() == 2

    db.store_function(
        db_session,
        function,
        "no_delete",
        project="project1",
        tag="latest",
        versioned=True,
    )
    db.delete_functions(db_session, "*", names=names[:2])

    assert db_session.query(Function.Label).count() == 3
    assert db_session.query(Function.Tag).count() == 5
    assert db_session.query(Function).count() == 3


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
