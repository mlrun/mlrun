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

import sqlalchemy.orm

import mlrun.config
import mlrun.utils.singleton
import server.api.utils.singletons.db


class Paginator(metaclass=mlrun.utils.singleton.AbstractSingleton):
    ttl = mlrun.config.config.pagination_cache.ttl
    max_size = mlrun.config.config.pagination_cache.max_size

    @classmethod
    def monitor_pagination_cache(cls, session: sqlalchemy.orm.Session):
        # query the pagination cache table, remove records that their last_accessed is older than ttl.
        # if the table is larger than max_size, remove the oldest records.
        db = server.api.utils.singletons.db.get_db()
        db.list_paginated_query_cache_record(session, cls.ttl, cls.max_size)
