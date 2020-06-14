import pytest
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_default_to_latest(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla'}
    function_name_1 = 'function_name_1'
    function_hash_key = db.store_function(db_session, function_1, function_name_1)
    assert function_hash_key is not None
    function_queried_without_tag = db.get_function(db_session, function_name_1)
    function_queried_without_tag_hash = function_queried_without_tag['metadata']['hash']
    assert function_hash_key == function_queried_without_tag_hash
    assert function_queried_without_tag['metadata']['tag'] == 'latest'
    function_queried_with_tag = db.get_function(
        db_session, function_name_1, tag='latest'
    )
    function_queried_without_tag_hash = function_queried_with_tag['metadata']['hash']
    assert function_queried_with_tag is not None
    assert function_queried_with_tag['metadata']['tag'] == 'latest'
    assert function_queried_without_tag_hash == function_queried_without_tag_hash


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_versioned(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla'}
    function_name_1 = 'function_name_1'
    function_hash_key = db.store_function(
        db_session, function_1, function_name_1, versioned=True
    )
    function_queried_without_hash_key = db.get_function(db_session, function_name_1)
    assert function_queried_without_hash_key is not None
    assert function_queried_without_hash_key['metadata']['tag'] == 'latest'

    # Verifying versioned function is queryable by hash_key
    function_queried_with_hash_key = db.get_function(
        db_session, function_name_1, hash_key=function_hash_key
    )
    function_queried_with_hash_key_hash = function_queried_with_hash_key['metadata'][
        'hash'
    ]
    assert function_queried_with_hash_key is not None
    assert function_queried_with_hash_key['metadata']['tag'] == ''
    assert function_queried_with_hash_key_hash == function_hash_key

    function_2 = {'bla': 'blabla', 'bla2': 'blabla2'}
    function_name_1 = 'function_name_1'
    db.store_function(db_session, function_2, function_name_1, versioned=True)
    functions = db.list_functions(db_session, function_name_1)

    # Verifying both versions of the functions were saved
    assert len(functions) == 2

    tagged_count = 0
    for function in functions:
        if function['metadata']['tag'] == 'latest':
            tagged_count += 1

    # but only one was tagged
    assert tagged_count == 1


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_store_function_not_versioned(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla'}
    function_name_1 = 'function_name_1'
    function_hash_key = db.store_function(
        db_session, function_1, function_name_1, versioned=False
    )
    function_result_1 = db.get_function(db_session, function_name_1)
    assert function_result_1 is not None
    assert function_result_1['metadata']['tag'] == 'latest'

    # not versioned so not queryable by hash key
    function_result_2 = db.get_function(
        db_session, function_name_1, hash_key=function_hash_key
    )
    assert function_result_2 is None

    function_2 = {'bla': 'blabla', 'bla2': 'blabla2'}
    db.store_function(db_session, function_2, function_name_1, versioned=False)
    functions = db.list_functions(db_session, function_name_1)

    # Verifying only the latest version was saved
    assert len(functions) == 1


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_get_function_by_hash_key(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla', 'status': {'bla': 'blabla'}}
    function_name_1 = 'function_name_1'
    function_hash_key = db.store_function(
        db_session, function_1, function_name_1, versioned=True
    )
    function_queried_without_hash_key = db.get_function(db_session, function_name_1)
    assert function_queried_without_hash_key is not None

    # Verifying function is queryable by hash_key
    function_queried_with_hash_key = db.get_function(
        db_session, function_name_1, hash_key=function_hash_key
    )
    assert function_queried_with_hash_key is not None

    # function queried by hash shouldn't have tag
    assert function_queried_without_hash_key['metadata']['tag'] == 'latest'
    assert function_queried_with_hash_key['metadata']['tag'] == ''


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_get_function_by_tag(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla', 'status': {'bla': 'blabla'}}
    function_name_1 = 'function_name_1'
    function_hash_key = db.store_function(
        db_session, function_1, function_name_1, versioned=True
    )
    function_queried_by_tag = db.get_function(db_session, function_name_1, tag='latest')
    function_queried_by_hash_key = db.get_function(
        db_session, function_name_1, hash_key=function_hash_key
    )
    function_not_queried_by_tag_hash = function_queried_by_hash_key['metadata']['hash']
    assert function_hash_key == function_not_queried_by_tag_hash

    # function not queried by tag shouldn't have status
    assert function_queried_by_tag['status'] is not None
    assert function_queried_by_hash_key['status'] is None


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_list_functions_no_tags(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla', 'status': {'bla': 'blabla'}}
    function_2 = {'bla2': 'blabla', 'status': {'bla': 'blabla'}}
    function_name_1 = 'function_name_1'

    # It is impossible to create a function without tag - only to create with a tag, and then tag another function with
    # the same tag
    tag = 'some_tag'
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
        if function['metadata']['hash'] == function_1_hash_key:
            assert function['metadata']['tag'] == ''
            assert function['status'] is None


@pytest.mark.parametrize(
    "db,db_session", [(db, db) for db in dbs], indirect=["db", "db_session"]
)
def test_list_functions_multiple_tags(db: DBInterface, db_session: Session):
    function_1 = {'bla': 'blabla', 'status': {'bla': 'blabla'}}
    function_name_1 = 'function_name_1'

    tags = ['some_tag', 'some_tag2', 'some_tag3']
    for tag in tags:
        db.store_function(
            db_session, function_1, function_name_1, tag=tag, versioned=True
        )
    functions = db.list_functions(db_session, function_name_1)
    assert len(functions) == len(tags)
    for function in functions:
        function_tag = function['metadata']['tag']
        tags.remove(function_tag)
    assert len(tags) == 0
