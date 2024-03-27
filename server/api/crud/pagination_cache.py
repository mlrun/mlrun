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

import sqlalchemy.orm

import mlrun.config
import mlrun.utils.singleton
import server.api.crud
import server.api.utils.singletons.db


class PaginatedMethods:
    _paginated_methods_map = {
        "list_runs": server.api.crud.Runs().list_runs,
        "list_artifacts": server.api.crud.Artifacts().list_artifacts,
        "list_functions": server.api.crud.Functions().list_functions,
    }

    def get_paginated_method(self, method_name):
        return self._paginated_methods_map.get(method_name)


class PaginationCache(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._cache_ttl = mlrun.config.config.pagination_cache.ttl
        self._table_max_size = mlrun.config.config.pagination_cache.max_size

    # TODO: add pagination CRUD methods

    @staticmethod
    def cleanup_pagination_cache(session: sqlalchemy.orm.Session):
        db = server.api.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(session).delete()

    def monitor_pagination_cache(self, session: sqlalchemy.orm.Session):
        # query the pagination cache table, remove records that their last_accessed is older than ttl.
        # if the table is larger than max_size, remove the oldest records.
        db = server.api.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(
            session, last_accessed_before=datetime.datetime.now() - self._cache_ttl
        ).delete()
        table_size = db.list_paginated_query_cache_record(
            session, order_by_desc=False
        ).count()
        if table_size > self._table_max_size:
            db.list_paginated_query_cache_record(session, order_by_desc=True).limit(
                table_size - self._table_max_size
            ).delete()
