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
import typing

import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.utils.singletons.db
from mlrun import mlconf


class PaginationCache(metaclass=mlrun.utils.singleton.Singleton):
    @staticmethod
    def store_pagination_cache_record(
        session: sqlalchemy.orm.Session,
        user: str,
        method: typing.Callable,
        current_page: int,
        page_size: int,
        kwargs: dict,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.store_paginated_query_cache_record(
            session, user, method.__name__, current_page, page_size, kwargs
        )

    @staticmethod
    def get_pagination_cache_record(session: sqlalchemy.orm.Session, key: str):
        db = server.api.utils.singletons.db.get_db()
        return db.get_paginated_query_cache_record(session, key)

    @staticmethod
    def list_pagination_cache_records(
        session: sqlalchemy.orm.Session,
        key: str = None,
        user: str = None,
        function: str = None,
        last_accessed_before: datetime = None,
        order_by: typing.Optional[
            mlrun.common.schemas.OrderType
        ] = mlrun.common.schemas.OrderType.desc,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.list_paginated_query_cache_record(
            session, key, user, function, last_accessed_before, order_by
        )

    @staticmethod
    def delete_pagination_cache_record(session: sqlalchemy.orm.Session, key: str):
        db = server.api.utils.singletons.db.get_db()
        db.delete_paginated_query_cache_record(session, key)

    @staticmethod
    def cleanup_pagination_cache(session: sqlalchemy.orm.Session):
        db = server.api.utils.singletons.db.get_db()
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

        db = server.api.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(
            session,
            last_accessed_before=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(seconds=cache_ttl),
            as_query=True,
        ).delete()

        all_records_query = db.list_paginated_query_cache_record(session, as_query=True)
        table_size = all_records_query.count()
        if table_size > table_max_size:
            records = all_records_query.limit(table_size - table_max_size)
            for record in records:
                session.delete(record)
            session.commit()
