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
#
import os

import dask.dataframe as dd

import mlrun.datastore
import mlrun.datastore.wasbfs


def test_http_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"
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
