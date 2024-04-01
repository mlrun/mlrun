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

import typing

import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.crud
import server.api.db.sqldb.models
import server.api.utils.pagination
from mlrun.utils import logger


def paginated_method(
    total_amount: int,
    page: typing.Optional[int] = None,
    page_size: typing.Optional[int] = None,
):
    items = [{"name": f"item{i}"} for i in range(total_amount)]
    if not page_size:
        return items

    page = page or 1
    if page < 1 or (page - 1) * page_size >= total_amount:
        raise StopIteration

    return items[(page - 1) * page_size : page * page_size]


@pytest.fixture()
def mock_paginated_method(monkeypatch):
    monkeypatch.setattr(
        server.api.utils.pagination.PaginatedMethods, "_methods", [paginated_method]
    )
    monkeypatch.setattr(
        server.api.utils.pagination.PaginatedMethods,
        "_method_map",
        {"paginated_method": paginated_method},
    )
    yield paginated_method


@pytest.fixture()
def cleanup_pagination_cache_on_teardown(db: sqlalchemy.orm.Session):
    yield
    server.api.crud.PaginationCache().cleanup_pagination_cache(db)


def test_paginated_method():
    """
    Test the above paginated_method function, which is used as a mock for the paginated methods
    in the following tests.
    """
    total_amount = 10
    page_size = 3

    items = paginated_method(total_amount, 1, page_size)
    assert len(items) == page_size
    assert items[0]["name"] == "item0"
    assert items[1]["name"] == "item1"
    assert items[2]["name"] == "item2"

    items = paginated_method(total_amount, 2, page_size)
    assert len(items) == page_size
    assert items[0]["name"] == "item3"
    assert items[1]["name"] == "item4"
    assert items[2]["name"] == "item5"

    items = paginated_method(total_amount, 3, page_size)
    assert len(items) == page_size
    assert items[0]["name"] == "item6"
    assert items[1]["name"] == "item7"
    assert items[2]["name"] == "item8"

    items = paginated_method(total_amount, 4, page_size)
    assert len(items) == 1
    assert items[0]["name"] == "item9"

    with pytest.raises(StopIteration):
        paginated_method(total_amount, 5, page_size)


def test_paginate_request(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination happy flow.
    Request paginated method with page and page size, and verify that the correct items are returned.
    Check the db for the pagination cache record.
    Continue requesting the next page until the end of the items. Meanwhile, check the db for the pagination
    cache record updates.
    Once response is empty, verify that the cache record was removed.
    """
    auth_info = mlrun.common.schemas.AuthInfo(user_id="user1")
    page_size = 3
    method_kwargs = {"total_amount": 5}

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Requesting first page")
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, auth_info, None, 1, page_size, **method_kwargs
    )
    _assert_paginated_response(
        response, pagination_info, 1, page_size, ["item0", "item1", "item2"]
    )

    logger.info("Checking db cache record")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, pagination_info["token"]
    )
    _assert_cache_record(
        cache_record, auth_info.user_id, paginated_method, 1, page_size
    )

    logger.info("Requesting second page")
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, auth_info, pagination_info["token"]
    )
    _assert_paginated_response(
        response, pagination_info, 2, page_size, ["item3", "item4"]
    )

    logger.info("Checking db cache record")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, pagination_info["token"]
    )
    _assert_cache_record(
        cache_record, auth_info.user_id, paginated_method, 2, page_size
    )

    logger.info("Saving token for next assert")
    token = pagination_info["token"]

    logger.info(
        "Requesting third page, which is the end of the items and should return empty response"
    )
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, auth_info, pagination_info["token"]
    )
    assert len(response) == 0
    assert not pagination_info

    logger.info("Checking db cache record was removed")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, token
    )
    assert cache_record is None


def test_paginate_other_users_token(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination with a token that was created by a different user.
    Request paginated method with page and page size, and verify that the correct items are returned.
    Check the db for the pagination cache record.
    Request the next page with the token, and with different user, and verify that a AccessDeniedError is raised.
    """
    auth_info_1 = mlrun.common.schemas.AuthInfo(user_id="user1")
    auth_info_2 = mlrun.common.schemas.AuthInfo(user_id="user2")
    page_size = 3
    method_kwargs = {"total_amount": 5}

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Requesting first page with user1")
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, auth_info_1, None, 1, page_size, **method_kwargs
    )
    _assert_paginated_response(
        response, pagination_info, 1, page_size, ["item0", "item1", "item2"]
    )

    logger.info("Checking db cache record")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, pagination_info["token"]
    )
    _assert_cache_record(
        cache_record, auth_info_1.user_id, paginated_method, 1, page_size
    )

    logger.info("Requesting second page with user2, should raise AccessDeniedError")
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        paginator.paginate_request(
            db, paginated_method, auth_info_2, pagination_info["token"]
        )

    logger.info(
        "Requesting second page without auth info, should raise AccessDeniedError"
    )
    with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
        paginator.paginate_request(db, paginated_method, None, pagination_info["token"])


def test_paginate_no_auth(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination with no auth info.
    Request paginated method without auth info, verify that the correct items are returned.
    Check the db for the pagination cache record.
    Request the next page with auth info of some user, and verify that the request is successful.
    """
    page_size = 3
    method_kwargs = {"total_amount": 5}

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Requesting first page")
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, None, None, 1, page_size, **method_kwargs
    )
    _assert_paginated_response(
        response, pagination_info, 1, page_size, ["item0", "item1", "item2"]
    )

    logger.info("Checking db cache record")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, pagination_info["token"]
    )
    _assert_cache_record(cache_record, None, paginated_method, 1, page_size)

    logger.info("Requesting second page with auth info of some user")
    auth_info = mlrun.common.schemas.AuthInfo(user_id="any-user")
    response, pagination_info = paginator.paginate_request(
        db, paginated_method, auth_info, pagination_info["token"]
    )
    _assert_paginated_response(
        response, pagination_info, 2, page_size, ["item3", "item4"]
    )

    logger.info("Checking db cache record")
    cache_record = server.api.crud.PaginationCache().get_pagination_cache_record(
        db, pagination_info["token"]
    )
    _assert_cache_record(
        cache_record, auth_info.user_id, paginated_method, 2, page_size
    )


def test_no_pagination(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination with no page and page size.
    Request paginated method with no page and page size, and verify that all items are returned.
    """
    auth_info = mlrun.common.schemas.AuthInfo(user_id="user1")
    method_kwargs = {"total_amount": 5}

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Requesting all items")
    response, pagination_info = paginator.paginate_request(
        db,
        paginated_method,
        auth_info,
        token=None,
        page=None,
        page_size=None,
        **method_kwargs,
    )
    assert len(response) == 5
    assert not pagination_info

    logger.info("Checking that no cache record was created")
    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 0


def test_pagination_not_supported(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination with a method that is not supported.
    Request a method that is not supported for pagination, and verify that a NotImplementedError is raised.
    """
    auth_info = mlrun.common.schemas.AuthInfo(user_id="user1")
    method_kwargs = {"total_amount": 5}

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Requesting a method that is not supported for pagination")
    with pytest.raises(NotImplementedError):
        paginator.paginate_request(
            db,
            lambda: paginated_method(5, 1, 3),
            auth_info,
            token=None,
            page=1,
            page_size=3,
            **method_kwargs,
        )


def test_pagination_cache_cleanup(
    mock_paginated_method,
    cleanup_pagination_cache_on_teardown,
    db: sqlalchemy.orm.Session,
):
    """
    Test pagination cache cleanup.
    Create paginated cache records and check that they are removed when calling cleanup_pagination_cache.
    """
    auth_info = mlrun.common.schemas.AuthInfo(user_id="user1")
    method_kwargs = {"total_amount": 5}
    page_size = 3
    token = None

    paginator = server.api.utils.pagination.Paginator()

    logger.info("Creating paginated cache records")
    for i in range(3):
        _, pagination_info = paginator.paginate_request(
            db,
            paginated_method,
            auth_info,
            None,
            1,
            page_size + i,
            **method_kwargs,
        )
        token = pagination_info["token"]

    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 3

    logger.info("Cleaning up pagination cache")
    paginator._pagination_cache.cleanup_pagination_cache(db)
    db.commit()

    logger.info("Checking that all records were removed")
    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 0

    logger.info("Try to get page with token")
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        paginator.paginate_request(
            db,
            paginated_method,
            auth_info,
            token,
            1,
            page_size,
            **method_kwargs,
        )


def _assert_paginated_response(
    response, pagination_info, page, page_size, expected_items
):
    assert len(response) == len(expected_items)
    for i, item in enumerate(expected_items):
        assert response[i]["name"] == item
    assert pagination_info["token"] is not None
    assert pagination_info["page"] == page
    assert pagination_info["page_size"] == page_size


def _assert_cache_record(
    cache_record: server.api.db.sqldb.models.PaginationCache,
    user: typing.Optional[str],
    method: typing.Callable,
    current_page: int,
    page_size: int,
):
    assert cache_record is not None
    assert cache_record.user == user
    assert cache_record.function == method.__name__
    assert cache_record.current_page == current_page
    assert cache_record.page_size == page_size
