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

import os
from typing import List, Optional

import storey


class NeedsMongoDBAccess:
    """Checks that params for access to MongoDB exist and are legal
    :param webapi:  URL tring connection to mongodb database.
                    If not set, the MONGODB_CONNECTION_STR environment variable will be used.
    """

    def __init__(self, webapi=None):
        _WEB_API_PREFIX = "mongodb+srv://"
        webapi = webapi or os.getenv("MONGODB_CONNECTION_STR")
        if not webapi:
            self._webapi_url = None
            print(
                "Missing webapi parameter or MONGODB_CONNECTION_STR environment variable. Using fakeredit instead"
            )
            return

        if not webapi.startswith(_WEB_API_PREFIX):
            webapi = f"{_WEB_API_PREFIX}{webapi}"

        self._webapi_url = webapi


class MongoDBDriver(NeedsMongoDBAccess, storey.Driver):
    """
    Database connection to MongoDB.
    :param webapi:  string connection to mongodb database.
                    If not set, the MONGODB_CONNECTION_STR environment variable will be used.
    """

    def __init__(
        self,
        key_prefix: str = None,
        webapi: Optional[str] = None,
        aggregation_attribute_prefix: str = "aggr_",
        aggregation_time_attribute_prefix: str = "_",
    ):

        NeedsMongoDBAccess.__init__(self, webapi)
        self._mongodb_client = None
        self._key_prefix = key_prefix if key_prefix else "storey: "
        self._mtime_name = "$_mtime_"
        self._storey_key = "storey_key"

        self._aggregation_attribute_prefix = aggregation_attribute_prefix
        self._aggregation_time_attribute_prefix = aggregation_time_attribute_prefix

    def _lazy_init(self):
        from pymongo import MongoClient

        self._closed = False
        if not self._mongodb_client:
            self._mongodb_client = MongoClient(self._webapi_url)

    def collection(self, container, table_path):
        return self._mongodb_client[container][table_path[1:].split("/")[0]]

    async def _save_schema(self, container, table_path, schema):
        self._lazy_init()
        return None

    async def _load_schema(self, container, table_path):
        self._lazy_init()
        return None

    async def _save_key(
        self, container, table_path, key, aggr_item, partitioned_by_key, additional_data
    ):
        self._lazy_init()
        mongodb_key = self.make_key(table_path, key)
        data = {}
        for key in additional_data.keys():
            data[key] = additional_data[key]
        data = dict(data, **{self._storey_key: mongodb_key})
        return self.collection(container, table_path).insert_one(data)

    async def _load_aggregates_by_key(self, container, table_path, key):
        self._lazy_init()
        mongodb_key = self.make_key(table_path, key)
        table_path = f"/{table_path[1:].split('/')[0]}"
        collection = self.collection(container, table_path)
        try:
            agg_val, values = await self._get_all_fields(mongodb_key, collection)
            if not agg_val:
                agg_val = None
            if not values:
                values = None
            return [agg_val, values]
        except Exception:
            return [None, None]

    def supports_aggregations(self):
        return False

    async def _load_by_key(self, container, table_path, key, attribute):
        self._lazy_init()
        mongodb_key = self.make_key(table_path, key)
        table_path = f"/{table_path[1:].split('/')[0]}"
        collection = self.collection(container, table_path)
        if attribute == "*":
            _, values = await self._get_all_fields(mongodb_key, collection)
        else:
            values = await self._get_specific_fields(mongodb_key, attribute, collection)
        return values

    async def close(self):
        pass

    def make_key(self, table_path, key):

        return "{}{}{}".format(self._key_prefix, table_path[1:].split("/")[0], key)

    async def _get_all_fields(self, mongodb_key: str, collection):

        try:
            response = collection.find_one(
                filter={self._storey_key: {"$eq": mongodb_key}}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get key {mongodb_key}. Response error was: {e}"
            )
        return None, {
            key: val
            for key, val in response.items()
            if key is not self._storey_key
            and not key.startswith(self._aggregation_attribute_prefix)
        }

    async def _get_specific_fields(
        self, mongodb_key: str, collection, attributes: List[str]
    ):
        try:
            response = collection.find_one(
                filter={self._storey_key: {"$eq": mongodb_key}}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get key {mongodb_key}. Response error was: {e}"
            )
        return {key: val for key, val in response.items() if key in attributes}
