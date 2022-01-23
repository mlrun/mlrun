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
import json
import os
from base64 import b64encode
from copy import copy
from datetime import datetime
from typing import Dict, List, Optional, Union

import v3io
from nuclio import KafkaTrigger
from nuclio.config import split_path

import mlrun
from mlrun.secrets import SecretsStore

from ..config import config
from ..model import DataSource
from ..platforms.iguazio import parse_v3io_path
from ..utils import get_class
from .utils import store_path_to_spark


def get_source_from_dict(source):
    kind = source.get("kind", "")
    if not kind:
        return None
    return source_kind_to_driver[kind].from_dict(source)


def get_source_step(source, key_fields=None, time_field=None, context=None):
    """initialize the source driver"""
    if hasattr(source, "to_csv"):
        source = DataFrameSource(source, context=context)
    if not key_fields and not source.key_field:
        raise mlrun.errors.MLRunInvalidArgumentError("key column is not defined")
    return source.to_step(key_fields, time_field, context)


class BaseSourceDriver(DataSource):
    support_spark = False
    support_storey = False

    def _get_store(self):
        store, _ = mlrun.store_manager.get_or_create_store(self.path)
        return store

    def to_step(self, key_field=None, time_field=None, context=None):
        import storey

        return storey.SyncEmitSource(context=context)

    def get_table_object(self):
        """get storey Table object"""
        return None

    def to_dataframe(self):
        return mlrun.store_manager.object(url=self.path).as_df()

    def filter_df_start_end_time(self, df):
        if self.start_time or self.end_time:
            self.start_time = (
                datetime.min if self.start_time is None else self.start_time
            )
            self.end_time = datetime.max if self.end_time is None else self.end_time
            df = df.filter(
                (df[self.time_field] > self.start_time)
                & (df[self.time_field] <= self.end_time)
            )
        return df

    def to_spark_df(self, session, named_view=False):
        if self.support_spark:
            df = session.read.load(**self.get_spark_options())
            if named_view:
                df.createOrReplaceTempView(self.name)
            return df
        raise NotImplementedError()

    def get_spark_options(self):
        # options used in spark.read.load(**options)
        raise NotImplementedError()

    def is_iterator(self):
        return False


class CSVSource(BaseSourceDriver):
    """
        Reads CSV file as input source for a flow.

        :parameter name: name of the source
        :parameter path: path to CSV file
        :parameter key_field: the CSV field to be used as the key for events. May be an int (field index) or string
            (field name) if with_header is True. Defaults to None (no key). Can be a list of keys.
        :parameter time_field: the CSV field to be parsed as the timestamp for events. May be an int (field index) or
            string (field name) if with_header is True. Defaults to None (no timestamp field). The field will be parsed
            from isoformat (ISO-8601 as defined in datetime.fromisoformat()). In case the format is not isoformat,
            timestamp_format (as defined in datetime.strptime()) should be passed in attributes.
        :parameter schedule: string to configure scheduling of the ingestion job.
        :parameter attributes: additional parameters to pass to storey. For example:
            attributes={"timestamp_format": '%Y%m%d%H'}
        :parameter parse_dates: Optional. List of columns (names or integers, other than time_field) that will be
            attempted to parse as date column.
        """

    kind = "csv"
    support_storey = True
    support_spark = True

    def __init__(
        self,
        name: str = "",
        path: str = None,
        attributes: Dict[str, str] = None,
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
        parse_dates: Optional[Union[List[int], List[str]]] = None,
    ):
        super().__init__(name, path, attributes, key_field, time_field, schedule)
        self._parse_dates = parse_dates

    def to_step(self, key_field=None, time_field=None, context=None):
        import storey

        attributes = self.attributes or {}
        if context:
            attributes["context"] = context
        return storey.CSVSource(
            paths=self.path,
            header=True,
            build_dict=True,
            key_field=self.key_field or key_field,
            time_field=self.time_field or time_field,
            storage_options=self._get_store().get_storage_options(),
            parse_dates=self._parse_dates,
            **attributes,
        )

    def get_spark_options(self):
        return {
            "path": store_path_to_spark(self.path),
            "format": "csv",
            "header": "true",
            "inferSchema": "true",
        }

    def to_dataframe(self):
        kwargs = self.attributes.get("reader_args", {})
        chunksize = self.attributes.get("chunksize")
        if chunksize:
            kwargs["chunksize"] = chunksize
        return mlrun.store_manager.object(url=self.path).as_df(
            parse_dates=self._parse_dates, **kwargs
        )

    def is_iterator(self):
        return True if self.attributes.get("chunksize") else False


class ParquetSource(BaseSourceDriver):
    """
       Reads Parquet file/dir as input source for a flow.

       :parameter name: name of the source
       :parameter path: path to Parquet file or directory
       :parameter key_field: the column to be used as the key for events. Can be a list of keys.
       :parameter time_field: the column to be parsed as the timestamp for events. Defaults to None
       :parameter start_filter: datetime. If not None, the results will be filtered by partitions and
            'filter_column' > start_filter. Default is None
       :parameter end_filter: datetime. If not None, the results will be filtered by partitions
            'filter_column' <= end_filter. Default is None
       :parameter filter_column: Optional. if not None, the results will be filtered by this column and
            start_filter & end_filter
       :parameter schedule: string to configure scheduling of the ingestion job. For example '*/30 * * * *' will
            cause the job to run every 30 minutes
       :parameter attributes: additional parameters to pass to storey.
    """

    kind = "parquet"
    support_storey = True
    support_spark = True

    def __init__(
        self,
        name: str = "",
        path: str = None,
        attributes: Dict[str, str] = None,
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
    ):

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        super().__init__(
            name,
            path,
            attributes,
            key_field,
            time_field,
            schedule,
            start_time,
            end_time,
        )

    def to_step(
        self,
        key_field=None,
        time_field=None,
        start_time=None,
        end_time=None,
        context=None,
    ):
        import storey

        attributes = self.attributes or {}
        if context:
            attributes["context"] = context
        return storey.ParquetSource(
            paths=self.path,
            key_field=self.key_field or key_field,
            time_field=self.time_field or time_field,
            storage_options=self._get_store().get_storage_options(),
            end_filter=self.end_time,
            start_filter=self.start_time,
            filter_column=self.time_field or time_field,
            **attributes,
        )

    def get_spark_options(self):
        return {
            "path": store_path_to_spark(self.path),
            "format": "parquet",
        }

    def to_dataframe(self):
        kwargs = self.attributes.get("reader_args", {})
        return mlrun.store_manager.object(url=self.path).as_df(
            format="parquet", **kwargs
        )


class BigQuerySource(BaseSourceDriver):
    """
       Reads Google BigQuery query results as input source for a flow.

       example::

            # use sql query
            query_string = "SELECT * FROM `the-psf.pypi.downloads20210328` LIMIT 5000"
            source = BigQuerySource("bq1", query=query_string,
                                    gcp_project="my_project",
                                    materialization_dataset="dataviews")

            # read a table
            source = BigQuerySource("bq2", table="the-psf.pypi.downloads20210328", gcp_project="my_project")


       :parameter name:  source name
       :parameter table: table name/path, cannot be used together with query
       :parameter query: sql query string
       :parameter materialization_dataset: for query with spark, The target dataset for the materialized view.
                                           This dataset should be in same location as the view or the queried tables.
                                           must be set to a dataset where the GCP user has table creation permission
       :parameter chunksize: number of rows per chunk (default large single chunk)
       :parameter key_field: the column to be used as the key for events. Can be a list of keys.
       :parameter time_field: the column to be parsed as the timestamp for events. Defaults to None
       :parameter schedule: string to configure scheduling of the ingestion job. For example '*/30 * * * *' will
            cause the job to run every 30 minutes
       :parameter gcp_project:  google cloud project name
       :parameter spark_options: additional spart read options
    """

    kind = "bigquery"
    support_storey = False
    support_spark = True

    def __init__(
        self,
        name: str = "",
        table: str = None,
        query: str = None,
        materialization_dataset: str = None,
        chunksize: int = None,
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
        gcp_project: str = None,
        spark_options: dict = None,
    ):
        if query and table:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "cannot specify both table and query args"
            )
        attrs = {
            "query": query,
            "table": table,
            "chunksize": chunksize,
            "gcp_project": gcp_project,
            "spark_options": spark_options,
            "materialization_dataset": materialization_dataset,
        }
        attrs = {key: value for key, value in attrs.items() if value is not None}
        super().__init__(
            name,
            attributes=attrs,
            key_field=key_field,
            time_field=time_field,
            schedule=schedule,
        )
        self._rows_iterator = None

    def _get_credentials_string(self):
        gcp_project = self.attributes.get("gcp_project", None)
        key = "GCP_CREDENTIALS"
        gcp_cred_string = os.getenv(key) or os.getenv(
            SecretsStore.k8s_env_variable_name_for_secret(key)
        )
        return gcp_cred_string, gcp_project

    def _get_credentials(self):
        from google.oauth2 import service_account

        gcp_cred_string, gcp_project = self._get_credentials_string()
        if gcp_cred_string and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            gcp_cred_dict = json.loads(gcp_cred_string, strict=False)
            credentials = service_account.Credentials.from_service_account_info(
                gcp_cred_dict
            )
            return credentials, gcp_project or gcp_cred_dict["project_id"]
        return None, gcp_project

    def to_dataframe(self):
        from google.cloud import bigquery

        credentials, gcp_project = self._get_credentials()
        bqclient = bigquery.Client(project=gcp_project, credentials=credentials)
        query_job = bqclient.query(self.attributes.get("query"))

        chunksize = self.attributes.get("chunksize")
        if chunksize:
            self._rows_iterator = query_job.result(page_size=chunksize)
            return self._rows_iterator.to_dataframe_iterable()

        self._rows_iterator = query_job.result(page_size=chunksize)
        return self._rows_iterator.to_dataframe()

    def is_iterator(self):
        return True if self.attributes.get("chunksize") else False

    def to_spark_df(self, session, named_view=False):
        options = copy(self.attributes.get("spark_options", {}))
        credentials, gcp_project = self._get_credentials_string()
        if credentials:
            options["credentials"] = b64encode(credentials.encode("utf-8")).decode(
                "utf-8"
            )
        if gcp_project:
            options["parentProject"] = gcp_project
        query = self.attributes.get("query")
        table = self.attributes.get("table")
        materialization_dataset = self.attributes.get("materialization_dataset")
        if not query and not table:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "table or query args must be specified"
            )
        if query and not materialization_dataset:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "materialization_dataset must be specified when running a query"
            )
        if query:
            options["viewsEnabled"] = True
            options["materializationDataset"] = materialization_dataset
            options["query"] = query
        if table:
            options["path"] = table

        df = session.read.format("bigquery").load(**options)
        if named_view:
            df.createOrReplaceTempView(self.name)
        return df


class CustomSource(BaseSourceDriver):
    kind = "custom"
    support_storey = True
    support_spark = False

    def __init__(
        self,
        class_name: str = None,
        name: str = "",
        schedule: str = None,
        **attributes,
    ):
        attributes = attributes or {}
        attributes["class_name"] = class_name
        super().__init__(name, "", attributes, schedule=schedule)

    def to_step(self, key_field=None, time_field=None, context=None):
        attributes = copy(self.attributes)
        class_name = attributes.pop("class_name")
        class_object = get_class(class_name)
        return class_object(context=context, **attributes)


class DataFrameSource:
    """
       Reads data frame as input source for a flow.

       :parameter key_field: the column to be used as the key for events. Can be a list of keys. Defaults to None
       :parameter time_field: the column to be parsed as the timestamp for events. Defaults to None
       :parameter context: MLRun context. Defaults to None
    """

    support_storey = True

    def __init__(
        self, df, key_field=None, time_field=None, context=None, iterator=False
    ):
        self._df = df
        if isinstance(key_field, str):
            self.key_field = [key_field]
        else:
            self.key_field = key_field
        self.time_field = time_field
        self.context = context
        self.iterator = iterator

    def to_step(self, key_field=None, time_field=None, context=None):
        import storey

        return storey.DataframeSource(
            dfs=self._df,
            key_field=self.key_field or key_field,
            time_field=self.time_field or time_field,
            context=self.context or context,
        )

    def to_dataframe(self):
        return self._df

    def is_iterator(self):
        return self.iterator


class OnlineSource(BaseSourceDriver):
    """online data source spec"""

    _dict_fields = [
        "kind",
        "name",
        "path",
        "attributes",
        "key_field",
        "time_field",
        "online",
        "workers",
    ]
    kind = ""

    def __init__(
        self,
        name: str = None,
        path: str = None,
        attributes: Dict[str, str] = None,
        key_field: str = None,
        time_field: str = None,
        workers: int = None,
    ):
        super().__init__(name, path, attributes, key_field, time_field)
        self.online = True
        self.workers = workers

    def to_step(self, key_field=None, time_field=None, context=None):
        import storey

        source_class = (
            storey.AsyncEmitSource
            if config.datastore.async_source_mode == "enabled"
            else storey.SyncEmitSource
        )
        source_args = self.attributes.get("source_args", {})

        src_class = source_class(
            context=context,
            key_field=self.key_field,
            time_field=self.time_field,
            full_event=True,
            **source_args,
        )

        return src_class

    def add_nuclio_trigger(self, function):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "This source type is not supported with ingestion service yet"
        )


class HttpSource(OnlineSource):
    kind = "http"

    def add_nuclio_trigger(self, function):
        trigger_args = self.attributes.get("trigger_args")
        if trigger_args:
            function.with_http(**trigger_args)
        return function


class StreamSource(OnlineSource):
    """Sets stream source for the flow. If stream doesn't exist it will create it"""

    kind = "v3ioStream"

    def __init__(
        self,
        name="stream",
        group="serving",
        seek_to="earliest",
        shards=1,
        retention_in_hours=24,
        extra_attributes: dict = None,
        **kwargs,
    ):
        """
           Sets stream source for the flow. If stream doesn't exist it will create it

           :param name: stream name. Default "stream"
           :param group: consumer group. Default "serving"
           :param seek_to: from where to consume the stream. Default earliest
           :param shards: number of shards in the stream. Default 1
           :param retention_in_hours: if stream doesn't exist and it will be created set retention time. Default 24h
           :param extra_attributes: additional nuclio trigger attributes (key/value dict)
        """
        attrs = {
            "group": group,
            "seek_to": seek_to,
            "shards": shards,
            "retention_in_hours": retention_in_hours,
            "extra_attributes": extra_attributes or {},
        }
        super().__init__(name, attributes=attrs, **kwargs)

    def add_nuclio_trigger(self, function):
        endpoint, stream_path = parse_v3io_path(self.path)
        v3io_client = v3io.dataplane.Client(endpoint=endpoint)
        container, stream_path = split_path(stream_path)
        res = v3io_client.create_stream(
            container=container,
            path=stream_path,
            shard_count=self.attributes["shards"],
            retention_period_hours=self.attributes["retention_in_hours"],
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        )
        res.raise_for_status([409, 204])
        function.add_v3io_stream_trigger(
            self.path,
            self.name,
            self.attributes["group"],
            self.attributes["seek_to"],
            self.attributes["shards"],
            extra_attributes=self.attributes.get("extra_attributes", {}),
        )
        return function


class KafkaSource(OnlineSource):
    """Sets kafka source for the flow"""

    kind = "kafka"

    def __init__(
        self,
        brokers="localhost:9092",
        topics="topic",
        group="serving",
        initial_offset="earliest",
        partitions=None,
        sasl_user=None,
        sasl_pass=None,
        **kwargs,
    ):
        """Sets kafka source for the flow

           :param brokers: list of broker IP addresses
           :param topics: list of topic names on which to listen.
           :param group: consumer group. Default "serving"
           :param initial_offset: from where to consume the stream. Default earliest
           :param partitions: Optional, A list of partitions numbers for which the function receives events.
           :param sasl_user: Optional, user name to use for sasl authentications
           :param sasl_pass: Optional, password to use for sasl authentications
        """
        if isinstance(topics, str):
            topics = [topics]
        if isinstance(brokers, str):
            brokers = [brokers]
        attrs = {
            "brokers": brokers,
            "topics": topics,
            "partitions": partitions,
            "group": group,
            "initial_offset": initial_offset,
        }
        if sasl_user and sasl_pass:
            attrs["sasl_user"] = sasl_user
            attrs["sasl_user"] = sasl_user
        super().__init__(attributes=attrs, **kwargs)

    def add_nuclio_trigger(self, function):
        partitions = self.attributes.get("partitions")
        trigger = KafkaTrigger(
            brokers=self.attributes["brokers"],
            topics=self.attributes["topics"],
            partitions=partitions,
            consumer_group=self.attributes["group"],
            initial_offset=self.attributes["initial_offset"],
        )
        func = function.add_trigger("kafka", trigger)
        sasl_user = self.attributes.get("sasl_user")
        sasl_pass = self.attributes.get("sasl_pass")
        if sasl_user and sasl_pass:
            trigger.sasl(sasl_user, sasl_pass)
        replicas = 1 if not partitions else len(partitions)
        func.spec.min_replicas = replicas
        func.spec.max_replicas = replicas
        return func


# map of sources (exclude DF source which is not serializable)
source_kind_to_driver = {
    "": BaseSourceDriver,
    CSVSource.kind: CSVSource,
    ParquetSource.kind: ParquetSource,
    HttpSource.kind: HttpSource,
    StreamSource.kind: StreamSource,
    KafkaSource.kind: KafkaSource,
    CustomSource.kind: CustomSource,
    BigQuerySource.kind: BigQuerySource,
}
