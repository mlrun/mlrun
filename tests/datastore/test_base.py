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
import tempfile
from contextlib import nullcontext as does_not_raise
from datetime import datetime
from unittest.mock import Mock

import dask.dataframe as dd
import pandas as pd
import pytest
import pytz

import mlrun.datastore
import mlrun.datastore.wasbfs
from mlrun import MLRunInvalidArgumentError, new_function
from mlrun.datastore import KafkaSource
from mlrun.datastore.azure_blob import AzureBlobStore
from mlrun.datastore.base import HttpStore
from mlrun.datastore.datastore import schema_to_store
from mlrun.datastore.datastore_profile import DatastoreProfileKafkaSource
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
        "s3://aws-public-blockchain/v1.0/btc/blocks/date=2023-02-27/"
        "part-00000-7de4c87e-242f-4568-b5d7-aae4cc75e9ad-c000.snappy.parquet"
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


def test_kafka_source_with_attributes_as_ds_profile():
    ds = DatastoreProfileKafkaSource(
        name="dskafkasrc",
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
        kwargs_public={
            "sasl": {
                "handshake": True,
            },
        },
        kwargs_private={
            "sasl": {
                "password": "wrong_password",
            },
        },
    )
    source = KafkaSource(path="ds://dskafkasrc")
    function = new_function(kind="remote")
    mlrun.datastore.sources.datastore_profile_read = Mock(return_value=ds)
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


def test_kafka_source_with_attributes_as_ds_profile_brokers_list():
    ds = DatastoreProfileKafkaSource(
        name="dskafkasrc",
        brokers=["broker_host:9092", "broker_host2:9093"],
        topics=["mytopic", "mytopic2"],
        group="mygroup",
        kwargs_public={
            "sasl": {
                "handshake": True,
                "enabled": True,
            },
        },
        kwargs_private={
            "sasl": {
                "password": "mypassword",
                "user": "myuser",
            },
        },
    )
    source = KafkaSource(path="ds://dskafkasrc")
    function = new_function(kind="remote")
    mlrun.datastore.sources.datastore_profile_read = Mock(return_value=ds)
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092", "broker_host2:9093"]
    assert attributes["topics"] == ["mytopic", "mytopic2"]
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


# ML-6308
@pytest.mark.parametrize("start_time_tz", [None, "naive", "with_tz"])
@pytest.mark.parametrize("end_time_tz", [None, "naive", "with_tz"])
@pytest.mark.parametrize("df_tz", [False, True])
def test_as_df_time_filters(start_time_tz, end_time_tz, df_tz):
    time_column = "timestamp"

    parquet_file = os.path.join(
        os.path.join(os.path.dirname(__file__), "assets"), "testdata.parquet"
    )
    full_df = pd.read_parquet(parquet_file)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".parquet", delete=True
    ) as parquet_file:
        if df_tz:
            full_df[time_column] = full_df[time_column].dt.tz_localize("UTC")
        full_df.to_parquet(parquet_file.name)

        data_item = mlrun.datastore.store_manager.object(f"file://{parquet_file.name}")

        start_time = None
        tzinfo = pytz.UTC if start_time_tz == "with_tz" else None
        if start_time_tz:
            start_time = datetime(2020, 12, 1, 17, 28, 15, tzinfo=tzinfo)

        end_time = None
        tzinfo = pytz.UTC if end_time_tz == "with_tz" else None
        if end_time_tz:
            end_time = datetime(2020, 12, 1, 17, 29, 15, tzinfo=tzinfo)

        if {start_time_tz, end_time_tz} == {"naive", "with_tz"}:
            expectation = pytest.raises(
                MLRunInvalidArgumentError,
                match="start_time and end_time must have the same time zone",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            resp = data_item.as_df(
                start_time=start_time, end_time=end_time, time_column=time_column
            )
            num_row_expected = (
                190 - (80 if start_time_tz else 0) - (90 if end_time_tz else 0)
            )
            assert len(resp) == num_row_expected
