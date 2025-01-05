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

import time

import pytest
import sqlalchemy.orm

import mlrun.errors
from mlrun import mlconf
from mlrun.utils import logger

import framework.utils.pagination_cache
import services.api.crud
from framework.db.sqldb.db import MAX_INT_32


def test_pagination_cache_monitor_ttl(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records with last_accessed time older than cache TTL, and check that they are removed
    when calling monitor_pagination_cache
    """
    ttl = 5
    mlconf.httpdb.pagination.pagination_cache.ttl = ttl

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating paginated cache records")
    for i in range(3):
        framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == 3
    )

    logger.debug(
        "Sleeping for cache TTL so that records will be removed in the monitor"
    )
    time.sleep(ttl + 2)

    logger.debug("Creating new paginated cache record that won't be expired")
    new_key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
        db, "user3", method, page, page_size, kwargs
    )

    logger.debug("Monitoring pagination cache")
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    logger.debug("Checking that old records were removed and new record still exists")
    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == 1
    )
    assert (
        framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
            db, new_key
        )
        is not None
    )


def test_pagination_cache_monitor_max_table_size(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records until the cache table reaches the max size, and check that the oldest records are
    removed when calling monitor_pagination_cache
    """
    max_size = 3
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating old paginated cache record")
    old_key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
        db, "user0", method, page, page_size, kwargs
    )

    logger.debug("Sleeping for 1 second to create time difference between records")
    time.sleep(1)

    logger.debug(
        "Creating paginated cache records up to max size (including the old record)"
    )
    for i in range(1, max_size):
        framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == max_size
    )

    logger.debug("Creating new paginated cache record to replace the old one")
    new_key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
        db, "user3", method, page, page_size, kwargs
    )

    logger.debug("Monitoring pagination cache")
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    logger.debug(
        "Checking that old record was removed and all other records still exist"
    )
    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == max_size
    )
    assert (
        framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
            db, new_key
        )
        is not None
    )
    assert (
        framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
            db, old_key
        )
        is None
    )


def test_pagination_cleanup(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records and check that they are removed when calling cleanup_pagination_cache
    """
    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating paginated cache records")
    for i in range(3):
        framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == 3
    )

    logger.debug("Cleaning up pagination cache")
    framework.utils.pagination_cache.PaginationCache().cleanup_pagination_cache(db)
    db.commit()

    logger.debug("Checking that all records were removed")
    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == 0
    )


@pytest.mark.parametrize(
    "page, page_size",
    [
        (MAX_INT_32 + 1, 100),  # page exceeds max allowed value
        (200, MAX_INT_32 + 1),  # page_size exceeds max allowed value
    ],
)
def test_store_paginated_query_cache_record_out_of_range(
    db: sqlalchemy.orm.Session, page: int, page_size: int
):
    method = services.api.crud.Projects().list_projects
    kwargs = {}

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, "user_name", method, page, page_size, kwargs
        )
