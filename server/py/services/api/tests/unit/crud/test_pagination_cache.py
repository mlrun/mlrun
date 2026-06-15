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
    ttl = 0.1
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

    # a minimum of 1.1 is required because `monitor_pagination_cache` adds 1 second buffer to the TTL check
    time.sleep(ttl + 1.1)

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


def test_pagination_cache_monitor_max_table_size_multiple_deletions(
    db: sqlalchemy.orm.Session,
):
    """
    Test that when table size exceeds max_size by multiple records, the subquery correctly
    deletes the oldest records (ordered by last_accessed ascending) in a single operation.
    """
    max_size = 5
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size
    mlconf.httpdb.pagination.pagination_cache.ttl = (
        3600  # Set high TTL to avoid TTL-based deletion
    )

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    # Create 8 records (3 more than max_size)
    # We'll create them with small delays to ensure different last_accessed times
    logger.debug("Creating multiple paginated cache records with time differences")
    keys = []
    for i in range(8):
        key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )
        keys.append(key)
        if i < 7:  # Don't sleep after the last one
            time.sleep(0.1)  # Small delay to ensure different timestamps

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == 8
    )

    # sanity, ensure keys are all unique
    assert len(keys) == len(set(keys))

    logger.debug("Monitoring pagination cache - should remove 3 oldest records")
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    # Should have exactly max_size records remaining
    remaining_records = framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
        db
    )
    assert len(remaining_records) == max_size

    # The 3 oldest records (first 3 created) should be deleted
    # The 5 newest records (last 5 created) should remain
    for i in range(3):  # First 3 should be deleted
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, keys[i]
            )
            is None
        )

    for i in range(3, 8):  # Last 5 should remain
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, keys[i]
            )
            is not None
        )


def test_pagination_cache_monitor_max_table_size_exact_match(
    db: sqlalchemy.orm.Session,
):
    """
    Test that when table_size exactly equals max_size, no records are deleted.
    """
    max_size = 5
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size
    mlconf.httpdb.pagination.pagination_cache.ttl = (
        3600  # Set high TTL to avoid TTL-based deletion
    )

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating exactly max_size records")
    keys = []
    for i in range(max_size):
        key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )
        keys.append(key)

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == max_size
    )

    logger.debug("Monitoring pagination cache - should not delete any records")
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    # All records should still exist
    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == max_size
    )

    for key in keys:
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, key
            )
            is not None
        )


def test_pagination_cache_monitor_max_table_size_large_excess(
    db: sqlalchemy.orm.Session,
):
    """
    Test that when table_size significantly exceeds max_size, the subquery correctly
    deletes all excess records in a single operation.
    """
    max_size = 3
    total_records = 10  # 7 records to delete
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size
    mlconf.httpdb.pagination.pagination_cache.ttl = (
        3600  # Set high TTL to avoid TTL-based deletion
    )

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug(f"Creating {total_records} records (max_size is {max_size})")
    keys = []
    for i in range(total_records):
        key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"user{i}", method, page, page_size, kwargs
        )
        keys.append(key)
        if i < total_records - 1:
            time.sleep(0.05)  # Small delay to ensure different timestamps

    assert (
        len(
            framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
                db
            )
        )
        == total_records
    )

    logger.debug(
        f"Monitoring pagination cache - should remove {total_records - max_size} oldest records"
    )
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    # Should have exactly max_size records remaining
    remaining_records = framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
        db
    )
    assert len(remaining_records) == max_size

    # The oldest records (first total_records - max_size) should be deleted
    # The newest records (last max_size) should remain
    deleted_count = total_records - max_size
    for i in range(deleted_count):
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, keys[i]
            )
            is None
        )

    for i in range(deleted_count, total_records):
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, keys[i]
            )
            is not None
        )


def test_pagination_cache_monitor_ttl_and_max_size_combined(
    db: sqlalchemy.orm.Session,
):
    """
    Test that monitor_pagination_cache correctly handles both TTL-based deletion
    and max_size-based deletion in a single call.
    """
    ttl = 0.1
    max_size = 5
    mlconf.httpdb.pagination.pagination_cache.ttl = ttl
    mlconf.httpdb.pagination.pagination_cache.max_size = max_size

    method = services.api.crud.Projects().list_projects
    page = 1
    page_size = 10
    kwargs = {}

    logger.debug("Creating old records that will expire by TTL")
    old_keys = []
    for i in range(3):
        key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"old_user{i}", method, page, page_size, kwargs
        )
        old_keys.append(key)

    # Wait for TTL to expire
    time.sleep(ttl + 1.1)

    logger.debug("Creating new records that exceed max_size")
    new_keys = []
    for i in range(max_size + 2):  # Create 2 more than max_size
        key = framework.utils.pagination_cache.PaginationCache().store_pagination_cache_record(
            db, f"new_user{i}", method, page, page_size, kwargs
        )
        new_keys.append(key)
        if i < max_size + 1:
            time.sleep(0.1)  # Small delay to ensure different timestamps

    total_before = len(
        framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
            db
        )
    )
    assert total_before == 3 + max_size + 2  # old + new records

    logger.debug(
        "Monitoring pagination cache - should remove TTL-expired and oldest new records"
    )
    framework.utils.pagination_cache.PaginationCache().monitor_pagination_cache(db)

    # Old records (expired by TTL) should be deleted
    for old_key in old_keys:
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, old_key
            )
            is None
        )

    # Should have exactly max_size new records remaining (oldest 2 new records deleted)
    remaining_records = framework.utils.pagination_cache.PaginationCache().list_pagination_cache_records(
        db
    )
    assert len(remaining_records) == max_size

    # The 2 oldest new records should be deleted
    assert (
        framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
            db, new_keys[0]
        )
        is None
    )
    assert (
        framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
            db, new_keys[1]
        )
        is None
    )

    # The newest max_size records should remain
    for i in range(2, len(new_keys)):
        assert (
            framework.utils.pagination_cache.PaginationCache().get_pagination_cache_record(
                db, new_keys[i]
            )
            is not None
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
