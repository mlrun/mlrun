# Copyright 2018 Iguazio
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

from typing import List

import storey


class SqlDBDriver(storey.Driver):
    """
    Database connection to Sql data basw.
    :param db_path: url string connection to sql database.
    :param primary_key: the primary key of the table.
    """

    def __init__(
        self,
        primary_key: str,
        db_path: str,
        aggregation_attribute_prefix: str = "aggr_",
        aggregation_time_attribute_prefix: str = "_",
    ):
        self._db_path = db_path
        self._sql_connection = None
        self._primary_key = primary_key
        self._mtime_name = "$_mtime_"
        self._storey_key = "storey_key"

        self._aggregation_attribute_prefix = aggregation_attribute_prefix
        self._aggregation_time_attribute_prefix = aggregation_time_attribute_prefix

    def _lazy_init(self):
        import sqlalchemy as db

        self._closed = False
        if not self._sql_connection:
            self._engine = db.create_engine(self._db_path)
            self._sql_connection = self._engine.connect()

    def table(self, table_path):
        import sqlalchemy as db

        metadata = db.MetaData()
        return db.Table(
            table_path[3:].split("/")[1],
            metadata,
            autoload=True,
            autoload_with=self._engine,
        )

    async def _save_schema(self, container, table_path, schema):
        self._lazy_init()
        return None

    async def _load_schema(self, container, table_path):
        self._lazy_init()
        return None

    async def _save_key(
        self, container, table_path, key, aggr_item, partitioned_by_key, additional_data
    ):
        import sqlalchemy as db

        self._lazy_init()

        table = self.table(table_path)
        return_val = None
        try:
            return_val = self._sql_connection.execute(
                table.insert(), [additional_data]
            )
        except db.exc.IntegrityError:
            pass
        return return_val

    async def _load_aggregates_by_key(self, container, table_path, key):
        self._lazy_init()
        table = self.table(table_path)
        try:
            agg_val, values = await self._get_all_fields(key, table)
            if not agg_val:
                agg_val = None
            if not values:
                values = None
            return [agg_val, values]
        except Exception:
            return [None, None]

    async def _load_by_key(self, container, table_path, key, attribute):
        self._lazy_init()
        table = self.table(table_path)
        if attribute == "*":
            _, values = await self._get_all_fields(key, table)
        else:
            values = None
        return values

    async def close(self):
        pass

    async def _get_all_fields(self, key, table):

        try:
            my_query = f"SELECT * FROM {table} where {self._primary_key}={key}"
            results = self._sql_connection.execute(my_query).fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to get key {key}. Response error was: {e}")

        return None, {
            results[0]._fields[i]: results[0][i] for i in range(len(results[0]))
        }

    async def _get_specific_fields(self, key: str, table, attributes: List[str]):
        try:
            my_query = f"SELECT {','.join(attributes)} FROM {table} where {self._primary_key}={key}"
            results = self._sql_connection.execute(my_query).fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to get key {key}. Response error was: {e}")

        return None, {
            results[0]._fields[i]: results[0][i] for i in range(len(results[0]))
        }

    def supports_aggregations(self):
        return False
