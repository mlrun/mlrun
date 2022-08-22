from datetime import datetime
import os
from typing import List, Optional, Union

import pandas
import storey
from storey import Event
from storey.dtypes import _termination_obj


class MongoDBSourceStorey(storey.sources._IterableSource, storey.sources.WithUUID):
    """Use mongodb collection as input source for a flow.

    :parameter key_field: column to be used as key for events. can be list of columns
    :parameter time_field: column to be used as time for events.
    :parameter id_field: column to be used as ID for events.
    :parameter db_name: the name of the database to access
    :parameter connection_string: your mongodb connection string
    :parameter collection_name: the name of the collection to access,
                                from the current database
    :parameter query: dictionary query for mongodb
    :parameter start_filter: If not None, the results will be filtered by partitions and 'filter_column' > start_filter.
                            Default is None
    :parameter end_filter:  If not None, the results will be filtered by partitions 'filter_column' <= end_filter.
                            Default is None

    """

    def __init__(
            self,
            db_name: str = None,
            connection_string: str = None,
            collection_name: str = None,
            query: dict = None,
            key_field: Union[str, List[str]] = None,
            start_filter: datetime = None,
            end_filter: datetime = None,
            time_field: str = None,
            id_field: str = None,
            **kwargs,
    ):
        if query is None:
            query = {}
        if time_field:
            time_query = {time_field: {}}
            if start_filter:
                time_query[time_field]["$gte"] = start_filter
            if end_filter:
                time_query[time_field]["$lt"] = end_filter
            if time_query[time_field]:
                query.update(time_query)

        if key_field is not None:
            kwargs["key_field"] = key_field
        if time_field is not None:
            kwargs["time_field"] = time_field
        if id_field is not None:
            kwargs["id_field"] = id_field
        storey.sources._IterableSource.__init__(self, **kwargs)
        storey.sources.WithUUID.__init__(self)

        if not all([db_name, collection_name, connection_string]):
            raise ValueError(
                "cannot specify without connection_string, db_name and collection_name args"
            )

        self.query = query
        self.collection_name = collection_name
        self.db_name = db_name
        self.connection_string = connection_string

        self._key_field = key_field
        if time_field:
            self._time_field = time_field.split(".")
        else:
            self._time_field = time_field
        if id_field:
            self._id_field = id_field.split(".")
        else:
            self._id_field = id_field

    async def _run_loop(self):
        from pymongo import MongoClient

        mongodb_client = MongoClient(self.connection_string)
        db = mongodb_client[self.db_name]
        collection = db[self.collection_name]
        cursor = collection.find(self.query)
        for body in cursor:
            create_event = True
            if "_id" in body.keys():
                body["_id"] = str(body["_id"])

            key = None
            if self._key_field:
                if isinstance(self._key_field, list):
                    key = []
                    for key_field in self._key_field:
                        if key_field not in body or pandas.isna(body[key_field]):
                            create_event = False
                            break
                        key.append(body[key_field])
                else:
                    key = body[self._key_field]
                    if key is None:
                        create_event = False
            if create_event:
                time = None
                if self._time_field:
                    time = self.get_val_from_multi_dictionary(body, self._time_field)
                if self._id_field:
                    _id = self.get_val_from_multi_dictionary(body, self._id_field)
                else:
                    _id = self._get_uuid()
                event = Event(body, key=key, time=time, id=_id)
                await self._do_downstream(event)
            else:
                if self.context:
                    self.context.logger.error(
                        f"For {body} value of key {key_field} is None"
                    )
        return await self._do_downstream(_termination_obj)

    def get_val_from_multi_dictionary(self, event, field):
        for f in field:
            event = event[f]

        return event


class NeedsMongoDBAccess:
    """Checks that params for access to Redis exist and are legal
    :param webapi: URL to the web API (https or http). If not set, the REDIS_URL environment variable will be used.
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
    """Abstract class for database connection"""

    def __init__(
            self,
            # redis_type: RedisType = RedisType.STANDALONE,
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


class SqlDBSourceStorey(storey.sources._IterableSource, storey.sources.WithUUID):
    """Use mongodb collection as input source for a flow.

        :parameter key_field: column to be used as key for events. can be list of columns
        :parameter time_field: column to be used as time for events.
        :parameter id_field: column to be used as ID for events.
        :parameter db_name: the name of the database to access
        :parameter connection_string: your mongodb connection string
        :parameter collection_name: the name of the collection to access,
                                    from the current database
        :parameter query: dictionary query for mongodb
        :parameter start_filter: If not None, the results will be filtered by partitions and 'filter_column' > start_filter.
                                Default is None
        :parameter end_filter:  If not None, the results will be filtered by partitions 'filter_column' <= end_filter.
                                Default is None

        """

    def __init__(
            self,
            db_path: str = None,
            collection_name: str = None,
            query: dict = None,
            key_field: Optional[Union[str, List[str]]] = None,
            start_filter: Optional[datetime] = None,
            end_filter: Optional[datetime] = None,
            time_field: Optional[str] = None,
            id_field: Optional[str] = None,
            **kwargs,
    ):
        # if query is None:
        #     query = {}
        # if time_field:
        #     time_query = {time_field: {}}
        #     if start_filter:
        #         time_query[time_field]["$gte"] = start_filter
        #     if end_filter:
        #         time_query[time_field]["$lt"] = end_filter
        #     if time_query[time_field]:
        #         query.update(time_query)

        if key_field is not None:
            kwargs["key_field"] = key_field
        if time_field is not None:
            kwargs["time_field"] = time_field
        if id_field is not None:
            kwargs["id_field"] = id_field
        storey.sources._IterableSource.__init__(self, **kwargs)
        storey.sources.WithUUID.__init__(self)

        if not all([db_path, collection_name]):
            raise ValueError(
                "cannot specify without db_path and collection_name args"
            )

        self.query = query
        self.collection_name = collection_name
        self.db_path = db_path

        self._key_field = key_field
        if isinstance(time_field, str) and '.' in time_field:
            self._time_field = time_field.split(".")
        else:
            self._time_field = time_field
        if isinstance(id_field, str) and '.' in id_field:
            self._id_field = id_field.split(".")
        else:
            self._id_field = id_field

    async def _run_loop(self):
        import sqlalchemy as db
        engine = db.create_engine(self.db_path)
        metadata = db.MetaData()
        connection = engine.connect()
        collection = db.Table(self.collection_name, metadata, autoload=True, autoload_with=engine)
        results = connection.execute(db.select([collection])).fetchall()
        df = pandas.DataFrame(results)
        df.columns = results[0].keys()
        df.set_index('index', inplace=True)
        connection.close()

        for namedtuple in df.itertuples():
            create_event = True
            body = namedtuple._asdict()
            body.pop('Index')
            # if len(df.index.names) > 1:
            #     for i, index_column in enumerate(df.index.names):
            #         body[index_column] = index[i]
            # elif df.index.names[0] is not None:
            #     body[df.index.names[0]] = index
            key = None
            if self._key_field:
                if isinstance(self._key_field, list):
                    key = []
                    for key_field in self._key_field:
                        if key_field not in body or pandas.isna(body[key_field]):
                            create_event = False
                            break
                        key.append(body[key_field])
                else:
                    key = body[self._key_field]
                    if key is None:
                        create_event = False
            if create_event:
                time = None
                if self._time_field:
                    time = self.get_val_from_multi_dictionary(body, self._time_field)
                if self._id_field:
                    _id = self.get_val_from_multi_dictionary(body, self._id_field)
                else:
                    _id = self._get_uuid()
                event = Event(body, key=key, time=time, id=_id)
                await self._do_downstream(event)
            else:
                if self.context:
                    self.context.logger.error(
                        f"For {body} value of key {key_field} is None"
                    )
        return await self._do_downstream(_termination_obj)

    def get_val_from_multi_dictionary(self, event, field):
        for f in field:
            event = event[f]

        return event


class SqlDBDriver(storey.Driver):
    """Abstract class for database connection"""

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
        return db.Table(table_path[3:].split('/')[1], metadata, autoload=True, autoload_with=self._engine)

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
            return_val = self._sql_connection.execute(collection.insert(), [
                additional_data
            ])
        except exc.IntegrityError:
            print(1)
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
            print(1)
            values = None
        return values

    async def close(self):
        pass

    async def _get_all_fields(self, key: str, collection):

        try:
            response = collection.select().where(collection.c[self._primary_key] == key)
            # result = self._sql_connection.execute(response)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get key {key}. Response error was: {e}"
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
        # try:
        #     response = collection.find_one(
        #         filter={self._storey_key: {"$eq": mongodb_key}}
        #     )
        # except Exception as e:
        #     raise RuntimeError(
        #         f"Failed to get key {mongodb_key}. Response error was: {e}"
        #     )
        # return {key: val for key, val in response.items() if key in attributes}
        pass

    def supports_aggregations(self):
        return False
