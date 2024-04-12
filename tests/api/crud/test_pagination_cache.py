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

import sqlalchemy.orm

import server.api.crud
from mlrun import mlconf
from mlrun.utils import logger


def test_pagination_cache_monitor_ttl(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records with last_accessed time older than cache TTL, and check that they are removed
    when calling monitor_pagination_cache
    """
    ttl = 5
    mlconf.httpdb.pagination.pagination_cache.ttl = ttl

    method = server.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating paginated cache records")
    for i in range(3):
        server.api.crud.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 3

    logger.debug(
        "Sleeping for cache TTL so that records will be removed in the monitor"
    )
    time.sleep(ttl + 2)

    logger.debug("Creating new paginated cache record that won't be expired")
    new_key = server.api.crud.PaginationCache().store_pagination_cache_record(
        db, "user3", method, page, page_size, kwargs
    )

    logger.debug("Monitoring pagination cache")
    server.api.crud.PaginationCache().monitor_pagination_cache(db)

    logger.debug("Checking that old records were removed and new record still exists")
    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 1
    assert (
        server.api.crud.PaginationCache().get_pagination_cache_record(db, new_key)
        is not None
    )


def test_pagination_cache_monitor_max_table_size(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records until the cache table reaches the max size, and check that the oldest records are
    removed when calling monitor_pagination_cache
    """
    max_size = 3
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size

    method = server.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating old paginated cache record")
    old_key = server.api.crud.PaginationCache().store_pagination_cache_record(
        db, "user0", method, page, page_size, kwargs
    )

    logger.debug("Sleeping for 1 second to create time difference between records")
    time.sleep(1)

    logger.debug(
        "Creating paginated cache records up to max size (including the old record)"
    )
    for i in range(1, max_size):
        server.api.crud.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert (
        len(server.api.crud.PaginationCache().list_pagination_cache_records(db))
        == max_size
    )

    logger.debug("Creating new paginated cache record to replace the old one")
    new_key = server.api.crud.PaginationCache().store_pagination_cache_record(
        db, "user3", method, page, page_size, kwargs
    )

    logger.debug("Monitoring pagination cache")
    server.api.crud.PaginationCache().monitor_pagination_cache(db)

    logger.debug(
        "Checking that old record was removed and all other records still exist"
    )
    assert (
        len(server.api.crud.PaginationCache().list_pagination_cache_records(db))
        == max_size
    )
    assert (
        server.api.crud.PaginationCache().get_pagination_cache_record(db, new_key)
        is not None
    )
    assert (
        server.api.crud.PaginationCache().get_pagination_cache_record(db, old_key)
        is None
    )


def test_pagination_cleanup(db: sqlalchemy.orm.Session):
    """
    Create paginated cache records and check that they are removed when calling cleanup_pagination_cache
    """
    method = server.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating paginated cache records")
    for i in range(3):
        server.api.crud.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )

    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 3

    logger.debug("Cleaning up pagination cache")
    server.api.crud.PaginationCache().cleanup_pagination_cache(db)
    db.commit()

    logger.debug("Checking that all records were removed")
    assert len(server.api.crud.PaginationCache().list_pagination_cache_records(db)) == 0
