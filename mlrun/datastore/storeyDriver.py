import os
from typing import List, Optional

import storey


class NeedsMongoDBAccess:
    """Checks that params for access to MongoDB exist and are legal
    :param webapi:  URL tring connection to mongodb database.
                    If not set, the MONGODB_CONNECTION_STR environment variable will be used.
    """

    def __init__(self, webapi=None):
        webapi = webapi or os.getenv("MONGODB_CONNECTION_STR")
        if not webapi:
            self._webapi_url = None
            print(
                "Missing webapi parameter or MONGODB_CONNECTION_STR environment variable. Using fakeredit instead"
            )
            return

        if not webapi.startswith("mongodb+srv://"):
            webapi = f"mongodb+srv://{webapi}"

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


class SqlDBDriver(storey.Driver):
    """
    Database connection to Sql data basw.
    :param db_path: url string connection to sql database.
    :param primary_key: the primary key of the collection.
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

    def collection(self, table_path):
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
        from sqlalchemy import exc

        self._lazy_init()

        collection = self.collection(table_path)
        return_val = None
        try:
            return_val = self._sql_connection.execute(
                collection.insert(), [additional_data]
            )
        except exc.IntegrityError:
            pass
        return return_val

    async def _load_aggregates_by_key(self, container, table_path, key):
        self._lazy_init()
        collection = self.collection(table_path)
        try:
            agg_val, values = await self._get_all_fields(key, collection)
            if not agg_val:
                agg_val = None
            if not values:
                values = None
            return [agg_val, values]
        except Exception:
            return [None, None]

    async def _load_by_key(self, container, table_path, key, attribute):
        self._lazy_init()
        collection = self.collection(table_path)
        if attribute == "*":
            _, values = await self._get_all_fields(key, collection)
        else:
            values = None
        return values

    async def close(self):
        pass

    async def _get_all_fields(self, key: str, collection):

        try:
            my_query = f"SELECT * FROM {collection} where {self._primary_key}={key}"
            results = self._sql_connection.execute(my_query).fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to get key {key}. Response error was: {e}")

        return None, {
            results[0]._fields[i]: results[0][i] for i in range(len(results[0]))
        }

    async def _get_specific_fields(self, key: str, collection, attributes: List[str]):
        try:
            my_query = f"SELECT {','.join(attributes)} FROM {collection} where {self._primary_key}={key}"
            results = self._sql_connection.execute(my_query).fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to get key {key}. Response error was: {e}")

        return None, {
            results[0]._fields[i]: results[0][i] for i in range(len(results[0]))
        }

    def supports_aggregations(self):
        return False
