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
import os
import string
from contextlib import nullcontext as does_not_raise

import dask.dataframe as dd
import pytest

import mlrun.datastore
import mlrun.datastore.wasbfs
from mlrun import new_function
from mlrun.datastore import KafkaSource
from mlrun.datastore.azure_blob import AzureBlobStore
from mlrun.datastore.base import HttpStore
from mlrun.datastore.datastore import schema_to_store
from mlrun.datastore.dbfs_store import DBFSStore
from mlrun.datastore.filestore import FileStore
from mlrun.datastore.google_cloud_storage import GoogleCloudStorageStore
from mlrun.datastore.redis import RedisStore
from mlrun.datastore.s3 import S3Store
from mlrun.datastore.v3io import V3ioStore


def test_http_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"
    )
    data_item.as_df()


def test_http_fs_parquet_with_params_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq?param1=1&param2=2"
    )
    data_item.as_df()


def test_s3_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "s3://aws-roda-hcls-datalake/gnomad/chrm/run-DataSink0-1-part-block-0-r-00009-snappy.parquet"
    )
    data_item.as_df()


def test_load_object_into_dask_dataframe():
    # Load a parquet file from Azure Open Datasets
    os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "azureopendatastorage"
    data_item = mlrun.datastore.store_manager.object(
        "az://tutorials/noaa_isd_weather/demo_data.parquet"
    )
    ddf = data_item.as_df(df_module=dd)
    assert isinstance(ddf, dd.DataFrame)


def test_load_object_into_dask_dataframe_using_wasbs_url():
    # Load a parquet file from Azure Open Datasets
    os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "azureopendatastorage"
    data_item = mlrun.datastore.store_manager.object(
        "wasbs://tutorials@dummyaccount/noaa_isd_weather/demo_data.parquet"
    )
    ddf = data_item.as_df(df_module=dd)
    assert isinstance(ddf, dd.DataFrame)


def test_kafka_source_with_attributes():
    source = KafkaSource(
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
        attributes={
            "sasl": {
                "handshake": True,
            },
        },
    )
    function = new_function(kind="remote")
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092"]
    assert attributes["topics"] == ["mytopic"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enabled": True,
        "user": "myuser",
        "password": "mypassword",
        "handshake": True,
    }


def test_kafka_source_without_attributes():
    source = KafkaSource(
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
    )
    function = new_function(kind="remote")
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092"]
    assert attributes["topics"] == ["mytopic"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enabled": True,
        "user": "myuser",
        "password": "mypassword",
    }


@pytest.mark.parametrize(
    "schemas,expected_class,expected",
    [
        (["file"] + list(string.ascii_lowercase), FileStore, does_not_raise()),
        (["s3"], S3Store, does_not_raise()),
        (["az", "wasbs", "wasb"], AzureBlobStore, does_not_raise()),
        (["v3io", "v3ios"], V3ioStore, does_not_raise()),
        (["redis", "rediss"], RedisStore, does_not_raise()),
        (["http", "https"], HttpStore, does_not_raise()),
        (["gcs", "gs"], GoogleCloudStorageStore, does_not_raise()),
        (["dbfs"], DBFSStore, does_not_raise()),
        (["random"], None, pytest.raises(ValueError)),
    ],
)
def test_schema_to_store(schemas, expected_class, expected):
    with expected:
        stores = [schema_to_store(schema) for schema in schemas]
        assert all(store == expected_class for store in stores)
