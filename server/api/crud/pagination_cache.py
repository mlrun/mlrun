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
import mlrun.config
import mlrun.utils.singleton
import server.api.crud
import server.api.utils.singletons.db


class PaginatedMethods:
    _paginated_methods = [
        server.api.crud.Projects().list_projects,
        server.api.crud.Artifacts().list_artifacts,
        server.api.crud.Functions().list_functions,
    ]
    _paginated_methods_map = {method.__name__: method for method in _paginated_methods}

    def get_paginated_method(self, method_name):
        return self._paginated_methods_map.get(method_name)


class PaginationCache(metaclass=mlrun.utils.singleton.Singleton):
    @staticmethod
    def store_pagination_cache_record(
        session: sqlalchemy.orm.Session,
        user: str,
        method: typing.Callable,
        current_page: int,
        kwargs: dict,
    ):
        db = server.api.utils.singletons.db.get_db()
        return db.store_paginated_query_cache_record(
            session, user, method.__name__, current_page, kwargs
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
        # query the pagination cache table, remove records that their last_accessed is older than ttl.
        # if the table is larger than max_size, remove the oldest records.
        cache_ttl = mlrun.config.config.httpdb.pagination_cache.ttl
        table_max_size = mlrun.config.config.httpdb.pagination_cache.max_size

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
                db.delete_paginated_query_cache_record(session, record.key)
