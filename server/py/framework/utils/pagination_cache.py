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

import datetime
import typing

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
from mlrun import mlconf

import framework.db.sqldb.models
import framework.utils.singletons.db


class PaginationCache(metaclass=mlrun.utils.singleton.Singleton):
    @staticmethod
    def store_pagination_cache_record(
        session: sqlalchemy.orm.Session,
        user: str,
        method: typing.Callable,
        current_page: int,
        page_size: int,
        kwargs: dict,
        pagination_cache_record: typing.Optional[
            framework.db.sqldb.models.PaginationCache
        ] = None,
    ):
        db = framework.utils.singletons.db.get_db()
        return db.store_paginated_query_cache_record(
            session,
            user,
            method.__name__,
            current_page,
            page_size,
            kwargs,
            pagination_cache_record,
        )

    @staticmethod
    def get_pagination_cache_record(
        session: sqlalchemy.orm.Session,
        key: str,
        for_update: bool = False,
    ) -> typing.Optional[framework.db.sqldb.models.PaginationCache]:
        """
        Retrieve a pagination cache record by its key.
        :param session: The database session to use for the operation.
        :param key: The unique key of the pagination cache record.
        :param for_update: Whether to lock the record for update.
        :return: The pagination cache record if found, else None.
        """
        db = framework.utils.singletons.db.get_db()
        return db.get_paginated_query_cache_record(session, key, for_update)

    @staticmethod
    def list_pagination_cache_records(
        session: sqlalchemy.orm.Session,
        key: typing.Optional[str] = None,
        user: typing.Optional[str] = None,
        function: typing.Optional[str] = None,
        last_accessed_before: typing.Optional[datetime.datetime] = None,
        order_by: typing.Optional[
            mlrun.common.schemas.OrderType
        ] = mlrun.common.schemas.OrderType.desc,
    ):
        db = framework.utils.singletons.db.get_db()
        return db.list_paginated_query_cache_record(
            session, key, user, function, last_accessed_before, order_by
        )

    @staticmethod
    def delete_pagination_cache_record(session: sqlalchemy.orm.Session, key: str):
        db = framework.utils.singletons.db.get_db()
        db.delete_paginated_query_cache_record(session, key)

    @staticmethod
    def cleanup_pagination_cache(session: sqlalchemy.orm.Session):
        db = framework.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(session, as_query=True).delete()

    @staticmethod
    def monitor_pagination_cache(session: sqlalchemy.orm.Session):
        """
        Monitor the pagination cache and remove records that are older than the cache TTL, and if the cache table
        reached the max size, remove the oldest records.
        """

        # Using cache TTL + 1 to make sure a zero TTL won't remove records that were just created
        cache_ttl = mlconf.httpdb.pagination.pagination_cache.ttl + 1
        table_max_size = mlconf.httpdb.pagination.pagination_cache.max_size

        db = framework.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(
            session,
            last_accessed_before=datetime.datetime.now(datetime.UTC)
            - datetime.timedelta(seconds=cache_ttl),
            as_query=True,
        ).delete()

        all_records_query = db.list_paginated_query_cache_record(session, as_query=True)
        table_size = all_records_query.count()
        if table_size > table_max_size:
            # Create a subquery to get the keys of the oldest records to delete
            # This executes as a single SQL DELETE with subquery, no Python iteration needed
            oldest_records_subquery = (
                db.list_paginated_query_cache_record(
                    session,
                    order_by=mlrun.common.schemas.OrderType.asc,
                    as_query=True,
                )
                .with_entities(framework.db.sqldb.models.PaginationCache.key)
                .limit(table_size - table_max_size)
            )

            # Delete records in a single SQL query using the subquery directly
            session.query(framework.db.sqldb.models.PaginationCache).filter(
                framework.db.sqldb.models.PaginationCache.key.in_(
                    oldest_records_subquery
                )
            ).delete(synchronize_session=False)
            session.commit()
