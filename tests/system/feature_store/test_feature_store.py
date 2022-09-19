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
import json
import math
import os
import pathlib
import random
import string
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from time import sleep

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import requests
from pandas.util.testing import assert_frame_equal
from storey import MapClass

import mlrun
import mlrun.feature_store as fs
import tests.conftest
from mlrun.config import config
from mlrun.data_types.data_types import InferOptions, ValueType
from mlrun.datastore.sources import (
    CSVSource,
    DataFrameSource,
    ParquetSource,
    StreamSource,
)
from mlrun.datastore.targets import (
    CSVTarget,
    KafkaTarget,
    NoSqlTarget,
    ParquetTarget,
    RedisNoSqlTarget,
    TargetTypes,
    get_target_driver,
)
from mlrun.feature_store import Entity, FeatureSet
from mlrun.feature_store.feature_set import aggregates_step
from mlrun.feature_store.feature_vector import FixedWindowType
from mlrun.feature_store.steps import FeaturesetValidator, OneHotEncoder
from mlrun.features import MinMaxValidator
from tests.system.base import TestMLRunSystem

from .data_sample import quotes, stocks, trades


class MyMap(MapClass):
    def __init__(self, multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self._multiplier = multiplier

    def do(self, event):
        event["xx"] = event["bid"] * self._multiplier
        event["zz"] = 9
        return event


def my_func(df):
    return df


def myfunc1(x, context=None):
    assert context is not None, "context is none"
    x = x.drop(columns=["exchange"])
    return x


def _generate_random_name():
    random_name = "".join([random.choice(string.ascii_letters) for i in range(10)])
    return random_name


kafka_brokers = os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_BROKERS")

kafka_topic = "kafka_integration_test"


@pytest.fixture()
def kafka_consumer():
    import kafka

    # Setup
    kafka_admin_client = kafka.KafkaAdminClient(bootstrap_servers=kafka_brokers)
    kafka_consumer = kafka.KafkaConsumer(
        kafka_topic,
        bootstrap_servers=kafka_brokers,
        auto_offset_reset="earliest",
    )
    try:
        kafka_admin_client.delete_topics([kafka_topic])
        sleep(1)
    except kafka.errors.UnknownTopicOrPartitionError:
        pass
    kafka_admin_client.create_topics([kafka.admin.NewTopic(kafka_topic, 1, 1)])

    # Test runs
    yield kafka_consumer

    # Teardown
    kafka_admin_client.delete_topics([kafka_topic])
    kafka_admin_client.close()
    kafka_consumer.close()


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestFeatureStore(TestMLRunSystem):
    project_name = "fs-system-test-project"

    def custom_setup(self):
        pass

    def _generate_vector(self):
        data = pd.DataFrame({"name": ["ab", "cd"], "data": [10, 20]})

        data.set_index(["name"], inplace=True)
        fset = fs.FeatureSet("pandass", entities=[fs.Entity("name")], engine="pandas")
        fs.ingest(featureset=fset, source=data)

        features = ["pandass.*"]
        vector = fs.FeatureVector("my-vec", features)
        return vector

    def _ingest_stocks_featureset(self):
        stocks_set = fs.FeatureSet(
            "stocks", entities=[Entity("ticker", ValueType.STRING)]
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(
                stocks_set,
                stocks,
                infer_options=fs.InferOptions.default(),
                run_config=fs.RunConfig(local=True),
            )

        df = fs.ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())

        self._logger.info(f"output df:\n{df}")
        stocks_set["name"].description = "some name"

        self._logger.info(f"stocks spec: {stocks_set.to_yaml()}")
        assert (
            stocks_set.spec.features["name"].description == "some name"
        ), "description was not set"
        assert len(df) == len(stocks), "dataframe size doesnt match"
        assert stocks_set.status.stats["exchange"], "stats not created"

    def _ingest_quotes_featureset(self):
        quotes_set = FeatureSet("stock-quotes", entities=["ticker"])

        flow = quotes_set.graph
        flow.to("MyMap", multiplier=3).to(
            "storey.Extend", _fn="({'z': event['bid'] * 77})"
        ).to("storey.Filter", "filter", _fn="(event['bid'] > 51.92)").to(
            FeaturesetValidator()
        )

        quotes_set.add_aggregation("ask", ["sum", "max"], "1h", "10m", name="asks1")
        quotes_set.add_aggregation("ask", ["sum", "max"], "5h", "10m", name="asks2")
        quotes_set.add_aggregation("bid", ["min"], "1h", "10m")

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            # no name parameter, different window
            quotes_set.add_aggregation("bid", ["max"], "5h", "10m")

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            # no name parameter, different period
            quotes_set.add_aggregation("bid", ["max"], "1h", "5m")

        quotes_set.add_aggregation(
            column="bid", operations=["max"], windows="1h", period="10m"
        )

        df = fs.preview(
            quotes_set,
            quotes,
            entity_columns=["ticker"],
            timestamp_key="time",
            options=fs.InferOptions.default(),
        )
        self._logger.info(f"quotes spec: {quotes_set.spec.to_yaml()}")
        assert df["zz"].mean() == 9, "map didnt set the zz column properly"
        quotes_set["bid"].validator = MinMaxValidator(min=52, severity="info")

        quotes_set.plot(
            str(self.results_path / "pipe.png"), rankdir="LR", with_targets=True
        )
        df = fs.ingest(quotes_set, quotes, return_df=True)
        self._logger.info(f"output df:\n{df}")
        assert quotes_set.status.stats.get("asks1_sum_1h"), "stats not created"

    def _get_offline_vector(self, features, features_size, engine=None):
        vector = fs.FeatureVector("myvector", features, "stock-quotes.xx")
        resp = fs.get_offline_features(
            vector, entity_rows=trades, entity_timestamp_column="time", engine=engine
        )
        assert len(vector.spec.features) == len(
            features
        ), "unexpected num of requested features"
        assert (
            len(vector.status.features) == features_size
        ), "unexpected num of returned features"
        assert (
            len(vector.status.stats) == features_size
        ), "unexpected num of feature stats"
        assert vector.status.label_column == "xx", "unexpected label_column name"

        df = resp.to_dataframe()
        columns = trades.shape[1] + features_size - 2  # - 2 keys
        assert df.shape[1] == columns, "unexpected num of returned df columns"
        resp.to_parquet(str(self.results_path / f"query-{engine}.parquet"))

        # check simple api without join with other df
        # test the use of vector uri
        vector.save()
        resp = fs.get_offline_features(vector.uri, engine=engine)
        df = resp.to_dataframe()
        assert df.shape[1] == features_size, "unexpected num of returned df columns"

    def _get_online_features(self, features, features_size):
        # test real-time query
        vector = fs.FeatureVector("my-vec", features)
        with fs.get_online_feature_service(vector) as svc:
            # check non existing column
            resp = svc.get([{"bb": "AAPL"}])

            # check that passing a dict (without list) works
            resp = svc.get({"ticker": "GOOG"})
            assert (
                resp[0]["name"] == "Alphabet Inc" and resp[0]["exchange"] == "NASDAQ"
            ), "unexpected online result"

            try:
                resp = svc.get("GOOG")
                assert False
            except mlrun.errors.MLRunInvalidArgumentError:
                pass

            # check passing a list of list (of entity values) works
            resp = svc.get([["GOOG"]])
            assert resp[0]["name"] == "Alphabet Inc", "unexpected online result"

            resp = svc.get([{"ticker": "a"}])
            assert resp[0] is None
            resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
            resp = svc.get([{"ticker": "AAPL"}])
            assert (
                resp[0]["name"] == "Apple Inc" and resp[0]["exchange"] == "NASDAQ"
            ), "unexpected online result"
            resp2 = svc.get([{"ticker": "AAPL"}], as_list=True)
            assert (
                len(resp2[0]) == features_size - 1
            ), "unexpected online vector size"  # -1 label

    def test_ingest_and_query(self):

        self._logger.debug("Creating stocks feature set")
        self._ingest_stocks_featureset()

        self._logger.debug("Creating stock-quotes feature set")
        self._ingest_quotes_featureset()

        self._logger.debug("Get offline feature vector")
        features = [
            "stock-quotes.bid",
            "stock-quotes.asks2_sum_5h",
            "stock-quotes.ask as mycol",
            "stocks.*",
        ]
        features_size = (
            len(features) + 1 + 1
        )  # (*) returns 2 features, label adds 1 feature

        # test fetch with the pandas merger engine
        self._get_offline_vector(features, features_size, engine="local")

        # test fetch with the dask merger engine
        self._get_offline_vector(features, features_size, engine="dask")

        self._logger.debug("Get online feature vector")
        self._get_online_features(features, features_size)

    def test_get_offline_features_with_or_without_indexes(self):
        # ingest test data
        par_target = ParquetTarget(
            **{
                "path": "v3io:///bigdata/system-test-project/parquet/",
                "name": "stocks-parquet",
            }
        )
        targets = [par_target]
        stocks_for_parquet = trades.copy()
        stocks_for_parquet["another_time"] = [
            pd.Timestamp("2021-03-28 13:30:00.023"),
            pd.Timestamp("2021-03-28 13:30:00.038"),
            pd.Timestamp("2021-03-28 13:30:00.048"),
            pd.Timestamp("2021-03-28 13:30:00.048"),
            pd.Timestamp("2021-03-28 13:30:00.048"),
        ]
        stocks_for_parquet.to_parquet(
            "v3io:///bigdata/system-test-project/stocks_test.parquet"
        )
        stocks_for_parquet.set_index("ticker", inplace=True)
        stocks_set = fs.FeatureSet(
            "stocks_parquet_test",
            "stocks set",
            [Entity("ticker", ValueType.STRING)],
            timestamp_key="time",
        )

        df = fs.ingest(stocks_set, stocks_for_parquet, targets)
        assert len(df) == len(stocks_for_parquet), "dataframe size doesnt match"

        # test get offline features with different parameters
        vector = fs.FeatureVector("offline-vec", ["stocks_parquet_test.*"])

        # with_indexes = False, entity_timestamp_column = None
        default_df = fs.get_offline_features(vector).to_dataframe()
        assert isinstance(
            default_df.index, pd.core.indexes.range.RangeIndex
        ), "index column is not of default type"
        assert default_df.index.name is None, "index column is not of default type"
        assert "time" not in default_df.columns, "'time' column shouldn't be present"
        assert (
            "ticker" not in default_df.columns
        ), "'ticker' column shouldn't be present"

        # with_indexes = False, entity_timestamp_column = "time"
        resp = fs.get_offline_features(vector, entity_timestamp_column="time")
        df_no_time = resp.to_dataframe()

        tmpdir = tempfile.mkdtemp()
        pq_path = f"{tmpdir}/features.parquet"
        resp.to_parquet(pq_path)
        read_back_df = pd.read_parquet(pq_path)
        assert read_back_df.equals(df_no_time)
        csv_path = f"{tmpdir}/features.csv"
        resp.to_csv(csv_path)
        read_back_df = pd.read_csv(csv_path, parse_dates=[2])
        assert read_back_df.equals(df_no_time)

        assert isinstance(
            df_no_time.index, pd.core.indexes.range.RangeIndex
        ), "index column is not of default type"
        assert df_no_time.index.name is None, "index column is not of default type"
        assert "time" not in df_no_time.columns, "'time' column should not be present"
        assert (
            "ticker" not in df_no_time.columns
        ), "'ticker' column shouldn't be present"
        assert (
            "another_time" in df_no_time.columns
        ), "'another_time' column should be present"

        # with_indexes = False, entity_timestamp_column = "invalid" - should return the timestamp column
        df_with_time = fs.get_offline_features(
            vector, entity_timestamp_column="another_time"
        ).to_dataframe()
        assert isinstance(
            df_with_time.index, pd.core.indexes.range.RangeIndex
        ), "index column is not of default type"
        assert df_with_time.index.name is None, "index column is not of default type"
        assert (
            "ticker" not in df_with_time.columns
        ), "'ticker' column shouldn't be present"
        assert "time" in df_with_time.columns, "'time' column should be present"
        assert (
            "another_time" not in df_with_time.columns
        ), "'another_time' column should not be present"

        vector.spec.with_indexes = True
        df_with_index = fs.get_offline_features(vector).to_dataframe()
        assert not isinstance(
            df_with_index.index, pd.core.indexes.range.RangeIndex
        ), "index column is of default type"
        assert df_with_index.index.name == "ticker"
        assert "time" in df_with_index.columns, "'time' column should be present"

    @pytest.mark.parametrize(
        "target_path, should_raise_error",
        [
            (None, False),  # default
            (f"v3io:///bigdata/{project_name}/gof_wt.parquet", False),  # single file
            (f"v3io:///bigdata/{project_name}/gof_wt/", False),  # directory
            (
                f"v3io:///bigdata/{project_name}/{{run_id}}/gof_wt.parquet",
                True,
            ),  # with run_id
        ],
    )
    def test_different_target_paths_for_get_offline_features(
        self, target_path, should_raise_error
    ):
        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fs.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        fs.ingest(stocks_set, stocks)

        vector = fs.FeatureVector("SjqevLXR", ["stocks_test.*"])
        target = ParquetTarget(name="parquet", path=target_path)
        if should_raise_error:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fs.get_offline_features(vector, with_indexes=True, target=target)
        else:
            fs.get_offline_features(vector, with_indexes=True, target=target)
            df = pd.read_parquet(target.get_target_path())
            assert df is not None

    @pytest.mark.parametrize(
        "should_succeed, is_parquet, is_partitioned, target_path",
        [
            # storey - csv - fail for directory
            (True, False, None, "v3io:///bigdata/dif-eng/file.csv"),
            (False, False, None, "v3io:///bigdata/dif-eng/csv"),
            # storey - parquet - fail for single file on partitioned
            (True, True, False, "v3io:///bigdata/dif-eng/pq"),
            (True, True, False, "v3io:///bigdata/dif-eng/file.pq"),
            (True, True, True, "v3io:///bigdata/dif-eng/pq"),
            (False, True, True, "v3io:///bigdata/dif-eng/file.pq"),
        ],
    )
    def test_different_paths_for_ingest_on_storey_engines(
        self, should_succeed, is_parquet, is_partitioned, target_path
    ):
        fset = FeatureSet("fsname", entities=[Entity("ticker")], engine="storey")
        target = (
            ParquetTarget(name="tar", path=target_path, partitioned=is_partitioned)
            if is_parquet
            else CSVTarget(name="tar", path=target_path)
        )
        if should_succeed:
            fs.ingest(fset, source=stocks, targets=[target])
            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.get_filesystem(False)
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fs.ingest(fset, source=stocks, targets=[target])

    @pytest.mark.parametrize(
        "should_succeed, is_parquet, is_partitioned, target_path, chunks",
        [
            (False, True, False, "v3io:///bigdata/pd-eng/pq", None),
            (True, True, False, "v3io:///bigdata/pd-eng/file.pq", None),
            (False, False, False, "v3io:///bigdata/pd-eng/csv", None),
            (True, False, False, "v3io:///bigdata/pd-eng/file.csv", None),
            (True, False, False, "v3io:///bigdata/pd-eng/csv", 100),
            (False, False, False, "v3io:///bigdata/pd-eng/file.csv", 100),
        ],
    )
    def test_different_paths_for_ingest_on_pandas_engines(
        self, should_succeed, is_parquet, is_partitioned, target_path, chunks
    ):
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )
        if chunks:
            source.attributes["chunksize"] = chunks
        fset = FeatureSet("pandaset", entities=[Entity("key")], engine="pandas")
        target = (
            ParquetTarget(name="tar", path=target_path, partitioned=is_partitioned)
            if is_parquet
            else CSVTarget(name="tar", path=target_path)
        )

        if should_succeed:
            fs.ingest(fset, source=source, targets=[target])
            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.get_filesystem(False)
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fs.ingest(fset, source=source, targets=[target])

    def test_nosql_no_path(self):
        df = pd.DataFrame(
            {
                "key": ["key1", "key2"],
                "time_stamp": [
                    datetime(2020, 11, 1, 17, 33, 15),
                    datetime(2020, 10, 1, 17, 33, 15),
                ],
                "another_time_column": [
                    datetime(2020, 9, 1, 17, 33, 15),
                    datetime(2020, 8, 1, 17, 33, 15),
                ],
            }
        )
        fset = fs.FeatureSet("nosql-no-path", entities=[Entity("time_stamp")])
        target_overwrite = True
        ingest_kw = dict()
        if target_overwrite is not None:
            ingest_kw["overwrite"] = target_overwrite
        fs.ingest(fset, df, infer_options=fs.InferOptions.default(), **ingest_kw)

        assert fset.status.targets[
            0
        ].get_path().get_absolute_path() == fset.get_target_path("parquet")
        assert fset.status.targets[
            1
        ].get_path().get_absolute_path() == fset.get_target_path("nosql")

    def test_feature_set_db(self):
        name = "stocks_test"
        stocks_set = fs.FeatureSet(name, entities=["ticker"])
        fs.preview(
            stocks_set,
            stocks,
        )
        stocks_set.save()
        db = mlrun.get_run_db()

        sets = db.list_feature_sets(self.project_name, name)
        assert len(sets) == 1, "bad number of results"

        feature_set = fs.get_feature_set(name, self.project_name)
        assert feature_set.metadata.name == name, "bad feature set response"

        fs.ingest(stocks_set, stocks)
        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
            fs.delete_feature_set(name, self.project_name)

        stocks_set.purge_targets()

        fs.delete_feature_set(name, self.project_name)
        sets = db.list_feature_sets(self.project_name, name)
        assert not sets, "Feature set should be deleted"

    def test_feature_vector_db(self):
        name = "fvec-test"
        fvec = fs.FeatureVector(name=name)

        db = mlrun.get_run_db()

        # TODO: Using to_dict due to a bug in httpdb api which will be fixed in another PR
        db.create_feature_vector(
            feature_vector=fvec.to_dict(), project=self.project_name
        )

        vecs = db.list_feature_vectors(self.project_name, name)
        assert len(vecs) == 1, "bad number of results"

        feature_vec = fs.get_feature_vector(name, self.project_name)
        assert feature_vec.metadata.name == name, "bad feature set response"

        fs.delete_feature_vector(name, self.project_name)
        vecs = db.list_feature_vectors(self.project_name, name)
        assert not vecs, "Feature vector should be deleted"

    def test_top_value_of_boolean_column(self):
        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fs.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        fs.ingest(stocks_set, stocks)

        vector = fs.FeatureVector("SjqevLXR", ["stocks_test.*"])
        fs.get_offline_features(vector)

        actual_stat = vector.get_stats_table().drop("hist", axis=1, errors="ignore")
        actual_stat = actual_stat.sort_index().sort_index(axis=1)
        assert isinstance(actual_stat["top"]["booly"], bool)

    def test_ingest_to_default_path(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )

        fs.ingest(
            measurements,
            source,
            infer_options=fs.InferOptions.schema() + fs.InferOptions.Stats,
            run_config=fs.RunConfig(local=True),
        )
        final_path = measurements.get_target_path()
        assert "latest" not in final_path
        assert measurements.status.targets is not None
        for target in measurements.status.targets:
            assert "latest" not in target.get_path().get_absolute_path()
            assert target.run_id is not None

    def test_serverless_ingest(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        target_path = os.path.relpath(str(self.results_path / "mycsv.csv"))
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )
        targets = [CSVTarget("mycsv", path=target_path)]
        if os.path.exists(target_path):
            os.remove(target_path)

        fs.ingest(
            measurements,
            source,
            targets,
            infer_options=fs.InferOptions.schema() + fs.InferOptions.Stats,
            run_config=fs.RunConfig(local=True),
        )
        final_path = measurements.get_target_path()
        assert os.path.exists(final_path), "result file was not generated"
        features = sorted(measurements.spec.features.keys())
        stats = sorted(measurements.status.stats.keys())
        print(features)
        print(stats)
        stats.remove("timestamp")
        stats.remove(key)
        assert features == stats, "didn't infer stats for all features"

    def test_non_partitioned_target_in_dir(self):
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )
        path = str(self.results_path / _generate_random_name())
        target = ParquetTarget(path=path, partitioned=False)

        fset = fs.FeatureSet(
            name="test", entities=[Entity("patient_id")], timestamp_key="timestamp"
        )
        fs.ingest(fset, source, targets=[target])

        path_with_runid = path + "/" + fset.status.targets[0].run_id

        list_files = os.listdir(path_with_runid)
        assert len(list_files) == 1 and not os.path.isdir(
            path_with_runid + "/" + list_files[0]
        )
        os.remove(path_with_runid + "/" + list_files[0])

    def test_ingest_with_timestamp(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = CSVSource(
            "mycsv",
            path=os.path.relpath(str(self.assets_path / "testdata.csv")),
            time_field="timestamp",
        )
        resp = fs.ingest(measurements, source)
        assert resp["timestamp"].head(n=1)[0] == datetime.fromisoformat(
            "2020-12-01 17:24:15.906352"
        )

    def test_csv_time_columns(self):
        df = pd.DataFrame(
            {
                "key": ["key1", "key2"],
                "time_stamp": [
                    datetime(2020, 11, 1, 17, 33, 15),
                    datetime(2020, 10, 1, 17, 33, 15),
                ],
                "another_time_column": [
                    datetime(2020, 9, 1, 17, 33, 15),
                    datetime(2020, 8, 1, 17, 33, 15),
                ],
            }
        )

        csv_path = tempfile.mktemp(".csv")
        df.to_csv(path_or_buf=csv_path, index=False)
        source = CSVSource(
            path=csv_path, time_field="time_stamp", parse_dates=["another_time_column"]
        )

        measurements = fs.FeatureSet(
            "fs", entities=[Entity("key")], timestamp_key="time_stamp"
        )
        try:
            resp = fs.ingest(measurements, source)
            df.set_index("key", inplace=True)
            assert_frame_equal(df, resp)
        finally:
            os.remove(csv_path)

    def test_featureset_column_types(self):
        data = pd.DataFrame(
            {
                "key": ["key1", "key2"],
                "str": ["my_string1", "my_string2"],
                "int": [123456, 234567],
                "float": [123.456, 234.567],
                "bool": [True, False],
                "timestamp": [
                    pd.Timestamp("1980-02-04 17:21:50.781"),
                    pd.Timestamp("2020-03-04 12:12:45.120"),
                ],
                "category": pd.Categorical(
                    ["a", "c"], categories=["d", "c", "b", "a"], ordered=True
                ),
            }
        )
        for key in data.keys():
            verify_ingest(data, key)
            verify_ingest(data, key, infer=True)

        # Timedelta isn't supported in parquet
        data["timedelta"] = pd.Timedelta("-1 days 2 min 3us")

        for key in ["key", "timedelta"]:
            verify_ingest(data, key, targets=[TargetTypes.nosql])
            verify_ingest(data, key, targets=[TargetTypes.nosql], infer=True)

    def test_filtering_parquet_by_time(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = ParquetSource(
            "myparquet",
            path=os.path.relpath(str(self.assets_path / "testdata.parquet")),
            time_field="timestamp",
            start_time=datetime(2020, 12, 1, 17, 33, 15),
            end_time="2020-12-01 17:33:16",
        )

        resp = fs.ingest(
            measurements,
            source,
            return_df=True,
        )
        assert len(resp) == 10

        # start time > timestamp in source
        source = ParquetSource(
            "myparquet",
            path=os.path.relpath(str(self.assets_path / "testdata.parquet")),
            time_field="timestamp",
            start_time=datetime(2022, 12, 1, 17, 33, 15),
            end_time="2022-12-01 17:33:16",
        )

        resp = fs.ingest(
            measurements,
            source,
            return_df=True,
        )
        assert len(resp) == 0

    @pytest.mark.parametrize("key_bucketing_number", [None, 0, 4])
    @pytest.mark.parametrize("partition_cols", [None, ["department"]])
    @pytest.mark.parametrize("time_partitioning_granularity", [None, "day"])
    def test_ingest_partitioned_by_key_and_time(
        self, key_bucketing_number, partition_cols, time_partitioning_granularity
    ):
        name = f"measurements_{uuid.uuid4()}"
        key = "patient_id"
        measurements = fs.FeatureSet(
            name, entities=[Entity(key)], timestamp_key="timestamp"
        )
        orig_columns = list(pd.read_csv(str(self.assets_path / "testdata.csv")).columns)
        source = CSVSource(
            "mycsv",
            path=os.path.relpath(str(self.assets_path / "testdata.csv")),
            time_field="timestamp",
        )
        measurements.set_targets(
            targets=[
                ParquetTarget(
                    partitioned=True,
                    key_bucketing_number=key_bucketing_number,
                    partition_cols=partition_cols,
                    time_partitioning_granularity=time_partitioning_granularity,
                )
            ],
            with_defaults=False,
        )

        resp1 = fs.ingest(measurements, source).to_dict()

        features = [
            f"{name}.*",
        ]
        vector = fs.FeatureVector("myvector", features)
        resp2 = fs.get_offline_features(
            vector, entity_timestamp_column="timestamp", with_indexes=True
        )
        resp2 = resp2.to_dataframe().to_dict()

        assert resp1 == resp2

        file_system = fsspec.filesystem("v3io")
        path = measurements.get_target_path("parquet")
        dataset = pq.ParquetDataset(
            path,
            filesystem=file_system,
        )
        partitions = [key for key, _ in dataset.pieces[0].partition_keys]

        if key_bucketing_number is None:
            expected_partitions = []
        elif key_bucketing_number == 0:
            expected_partitions = ["key"]
        else:
            expected_partitions = [f"hash{key_bucketing_number}_key"]
        expected_partitions += partition_cols or []
        if all(
            value is None
            for value in [
                key_bucketing_number,
                partition_cols,
                time_partitioning_granularity,
            ]
        ):
            time_partitioning_granularity = (
                mlrun.utils.helpers.DEFAULT_TIME_PARTITIONING_GRANULARITY
            )
        if time_partitioning_granularity:
            for unit in ["year", "month", "day", "hour"]:
                expected_partitions.append(unit)
                if unit == time_partitioning_granularity:
                    break

        assert partitions == expected_partitions

        resp = fs.get_offline_features(
            vector,
            start_time=datetime(2020, 12, 1, 17, 33, 15),
            end_time="2020-12-01 17:33:16",
            entity_timestamp_column="timestamp",
        )
        resp2 = resp.to_dataframe()
        assert len(resp2) == 10
        result_columns = list(resp2.columns)
        orig_columns.remove("patient_id")
        assert result_columns.sort() == orig_columns.sort()

    def test_ingest_twice_with_nulls(self):
        name = f"test_ingest_twice_with_nulls_{uuid.uuid4()}"
        key = "key"

        measurements = fs.FeatureSet(
            name, entities=[Entity(key)], timestamp_key="my_time"
        )
        columns = [key, "my_string", "my_time"]
        df = pd.DataFrame(
            [["mykey1", "hello", pd.Timestamp("2019-01-26 14:52:37")]], columns=columns
        )
        df.set_index("my_string")
        source = DataFrameSource(df)
        measurements.set_targets(
            targets=[ParquetTarget(partitioned=True)],
            with_defaults=False,
        )
        resp1 = fs.ingest(measurements, source)
        assert resp1.to_dict() == {
            "my_string": {"mykey1": "hello"},
            "my_time": {"mykey1": pd.Timestamp("2019-01-26 14:52:37")},
        }

        features = [
            f"{name}.*",
        ]
        vector = fs.FeatureVector("myvector", features)
        resp2 = fs.get_offline_features(vector, with_indexes=True)
        resp2 = resp2.to_dataframe()
        assert resp2.to_dict() == {
            "my_string": {"mykey1": "hello"},
            "my_time": {"mykey1": pd.Timestamp("2019-01-26 14:52:37")},
        }

        measurements = fs.FeatureSet(
            name, entities=[Entity(key)], timestamp_key="my_time"
        )
        columns = [key, "my_string", "my_time"]
        df = pd.DataFrame(
            [["mykey2", None, pd.Timestamp("2019-01-26 14:52:37")]], columns=columns
        )
        df.set_index("my_string")
        source = DataFrameSource(df)
        measurements.set_targets(
            targets=[ParquetTarget(partitioned=True)],
            with_defaults=False,
        )
        resp1 = fs.ingest(measurements, source, overwrite=False)
        assert resp1.to_dict() == {
            "my_string": {"mykey2": None},
            "my_time": {"mykey2": pd.Timestamp("2019-01-26 14:52:37")},
        }

        features = [
            f"{name}.*",
        ]
        vector = fs.FeatureVector("myvector", features)
        vector.spec.with_indexes = True
        resp2 = fs.get_offline_features(vector)
        resp2 = resp2.to_dataframe()
        assert resp2.to_dict() == {
            "my_string": {"mykey1": "hello", "mykey2": None},
            "my_time": {
                "mykey1": pd.Timestamp("2019-01-26 14:52:37"),
                "mykey2": pd.Timestamp("2019-01-26 14:52:37"),
            },
        }

    def test_ordered_pandas_asof_merge(self):
        targets = [ParquetTarget(), NoSqlTarget()]
        left_set, left = prepare_feature_set(
            "left", "ticker", trades, timestamp_key="time", targets=targets
        )
        right_set, right = prepare_feature_set(
            "right", "ticker", quotes, timestamp_key="time", targets=targets
        )

        features = ["left.*", "right.*"]
        feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
        res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
        res = res.to_dataframe()
        assert res.shape[0] == left.shape[0]

    def test_left_not_ordered_pandas_asof_merge(self):
        left = trades.sort_values(by="price")

        left_set, left = prepare_feature_set(
            "left", "ticker", left, timestamp_key="time"
        )
        right_set, right = prepare_feature_set(
            "right", "ticker", quotes, timestamp_key="time"
        )

        features = ["left.*", "right.*"]
        feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
        res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
        res = res.to_dataframe()
        assert res.shape[0] == left.shape[0]

    def test_right_not_ordered_pandas_asof_merge(self):
        right = quotes.sort_values(by="bid")

        left_set, left = prepare_feature_set(
            "left", "ticker", trades, timestamp_key="time"
        )
        right_set, right = prepare_feature_set(
            "right", "ticker", right, timestamp_key="time"
        )

        features = ["left.*", "right.*"]
        feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
        res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
        res = res.to_dataframe()
        assert res.shape[0] == left.shape[0]

    def test_read_csv(self):
        from storey import CSVSource, ReduceToDataFrame, build_flow

        csv_path = str(self.results_path / _generate_random_name() / ".csv")
        targets = [CSVTarget("mycsv", path=csv_path)]
        stocks_set = fs.FeatureSet(
            "tests", entities=[Entity("ticker", ValueType.STRING)]
        )
        fs.ingest(
            stocks_set, stocks, infer_options=fs.InferOptions.default(), targets=targets
        )

        # reading csv file
        final_path = stocks_set.get_target_path("mycsv")
        controller = build_flow([CSVSource(final_path), ReduceToDataFrame()]).run()
        termination_result = controller.await_termination()

        expected = pd.DataFrame(
            {
                0: ["ticker", "MSFT", "GOOG", "AAPL"],
                1: ["name", "Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                2: ["exchange", "NASDAQ", "NASDAQ", "NASDAQ"],
            }
        )

        assert termination_result.equals(
            expected
        ), f"{termination_result}\n!=\n{expected}"
        os.remove(final_path)

    def test_multiple_entities(self):
        name = f"measurements_{uuid.uuid4()}"
        current_time = pd.Timestamp.now()
        data = pd.DataFrame(
            {
                "time": [
                    current_time,
                    current_time - pd.Timedelta(minutes=1),
                    current_time - pd.Timedelta(minutes=2),
                    current_time - pd.Timedelta(minutes=3),
                    current_time - pd.Timedelta(minutes=4),
                    current_time - pd.Timedelta(minutes=5),
                ],
                "first_name": ["moshe", None, "yosi", "yosi", "moshe", "yosi"],
                "last_name": ["cohen", "levi", "levi", "levi", "cohen", "levi"],
                "bid": [2000, 10, 11, 12, 2500, 14],
            }
        )

        # write to kv
        data_set = fs.FeatureSet(
            name, entities=[Entity("first_name"), Entity("last_name")]
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )
        fs.preview(
            data_set,
            data,  # source
            entity_columns=["first_name", "last_name"],
            timestamp_key="time",
            options=fs.InferOptions.default(),
        )

        data_set.plot(
            str(self.results_path / "pipe.png"), rankdir="LR", with_targets=True
        )
        fs.ingest(data_set, data, return_df=True)

        features = [
            f"{name}.bid_sum_1h",
        ]

        vector = fs.FeatureVector("my-vec", features)
        with fs.get_online_feature_service(vector) as svc:
            resp = svc.get([{"first_name": "yosi", "last_name": "levi"}])
            assert resp[0]["bid_sum_1h"] == 37.0

    def test_time_with_timezone(self):
        data = pd.DataFrame(
            {
                "time": [
                    datetime(2021, 6, 30, 15, 9, 35, tzinfo=timezone.utc),
                    datetime(2021, 6, 30, 15, 9, 35, tzinfo=timezone.utc),
                ],
                "first_name": ["katya", "dina"],
                "bid": [2000, 10],
            }
        )
        data_set = fs.FeatureSet("fs4", entities=[Entity("first_name")])

        df = fs.ingest(data_set, data, return_df=True)

        data.set_index("first_name", inplace=True)
        assert_frame_equal(df, data)

    def test_offline_features_filter_non_partitioned(self):
        data = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2021-06-09 09:30:06.008"),
                    pd.Timestamp("2021-06-09 10:29:07.009"),
                    pd.Timestamp("2021-06-09 09:29:08.010"),
                ],
                "data": [10, 20, 30],
                "string": ["ab", "cd", "ef"],
            }
        )
        data_set1 = fs.FeatureSet("fs1", entities=[Entity("string")])
        targets = [ParquetTarget(partitioned=False), NoSqlTarget()]
        fs.ingest(
            data_set1, data, targets=targets, infer_options=fs.InferOptions.default()
        )
        features = ["fs1.*"]
        vector = fs.FeatureVector("vector", features)
        vector.spec.with_indexes = True

        resp = fs.get_offline_features(
            vector,
            entity_timestamp_column="time_stamp",
            start_time="2021-06-09 09:30",
            end_time=datetime(2021, 6, 9, 10, 30),
        )

        expected = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2021-06-09 09:30:06.008"),
                    pd.Timestamp("2021-06-09 10:29:07.009"),
                ],
                "data": [10, 20],
                "string": ["ab", "cd"],
            }
        )
        expected.set_index(keys="string", inplace=True)

        assert expected.equals(resp.to_dataframe())

    def test_filter_offline_multiple_featuresets(self):
        data = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2021-06-09 09:30:06.008"),
                    pd.Timestamp("2021-06-09 10:29:07.009"),
                    pd.Timestamp("2021-06-09 09:29:08.010"),
                ],
                "data": [10, 20, 30],
                "string": ["ab", "cd", "ef"],
            }
        )

        data_set1 = fs.FeatureSet("fs1", entities=[Entity("string")])
        targets = [ParquetTarget(partitioned=False), NoSqlTarget()]
        fs.ingest(
            data_set1, data, targets=targets, infer_options=fs.InferOptions.default()
        )

        data2 = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2021-07-09 09:30:06.008"),
                    pd.Timestamp("2021-07-09 10:29:07.009"),
                    pd.Timestamp("2021-07-09 09:29:08.010"),
                ],
                "data": [10, 20, 30],
                "string": ["ab", "cd", "ef"],
            }
        )

        data_set2 = fs.FeatureSet("fs2", entities=[Entity("string")])
        fs.ingest(data_set2, data2, infer_options=fs.InferOptions.default())

        features = ["fs2.data", "fs1.time_stamp"]

        vector = fs.FeatureVector("vector", features)
        resp = fs.get_offline_features(
            vector,
            entity_timestamp_column="time_stamp",
            start_time=datetime(2021, 6, 9, 9, 30),
            end_time=None,  # will translate to now()
        )
        assert len(resp.to_dataframe()) == 2

    def test_unaggregated_columns(self):
        test_base_time = datetime(2020, 12, 1, 17, 33, 15)

        data = pd.DataFrame(
            {
                "time": [test_base_time, test_base_time - pd.Timedelta(minutes=1)],
                "first_name": ["moshe", "yosi"],
                "last_name": ["cohen", "levi"],
                "bid": [2000, 10],
            }
        )

        name = f"measurements_{uuid.uuid4()}"

        # write to kv
        data_set = fs.FeatureSet(name, entities=[Entity("first_name")])

        data_set.add_aggregation(
            name="bids",
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        fs.ingest(data_set, data, return_df=True)

        features = [f"{name}.bids_sum_1h", f"{name}.last_name"]

        vector = fs.FeatureVector("my-vec", features)
        with fs.get_online_feature_service(vector) as svc:
            resp = svc.get([{"first_name": "moshe"}])
            expected = {"bids_sum_1h": 2000.0, "last_name": "cohen"}
            assert resp[0] == expected

    _split_graph_expected_default = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
            ],
            "ticker": ["GOOG", "GOOG", "AAPL", "GOOG"],
            "bid": [720.50, 720.50, 97.99, 720.50],
            "ask": [720.93, 720.93, 98.01, 720.88],
            "xx": [2161.50, 2161.50, 293.97, 2161.50],
            "zz": [9, 9, 9, 9],
            "extra": [55478.50, 55478.50, 7545.23, 55478.50],
        }
    )

    _split_graph_expected_side = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
            "extra2": [
                12248.50,
                883.15,
                883.49,
                883.83,
                12248.50,
                1665.83,
                12248.50,
                884.17,
            ],
        }
    )

    @pytest.mark.parametrize("engine", ["pandas", "storey", None])
    def test_ingest_default_targets_for_engine(self, engine):
        data = pd.DataFrame({"name": ["ab", "cd"], "data": [10, 20]})

        data.set_index(["name"], inplace=True)
        fs_name = f"{engine}fs"
        fset = fs.FeatureSet(fs_name, entities=[fs.Entity("name")], engine=engine)
        fs.ingest(featureset=fset, source=data)

        features = [f"{fs_name}.*"]
        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)
        try:
            resp = svc.get([{"name": "ab"}])
            assert resp[0] == {"data": 10}
        finally:
            svc.close()

    @pytest.mark.parametrize("partitioned", [True, False])
    def test_schedule_on_filtered_by_time(self, partitioned):
        name = f"sched-time-{str(partitioned).lower()}"

        now = datetime.now() + timedelta(minutes=2)
        data = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                    pd.Timestamp("2021-01-10 11:00:00"),
                ],
                "first_name": ["moshe", "yosi"],
                "data": [2000, 10],
            }
        )
        # writing down a remote source
        data_target = ParquetTarget(partitioned=False)
        data_set = fs.FeatureSet("sched_data", entities=[Entity("first_name")])
        fs.ingest(data_set, data, targets=[data_target])

        path = data_set.status.targets[0].path.format(
            run_id=data_set.status.targets[0].run_id
        )
        assert path == data_set.get_target_path()

        # the job will be scheduled every minute
        cron_trigger = "*/1 * * * *"

        source = ParquetSource(
            "myparquet", path=path, time_field="time", schedule=cron_trigger
        )

        feature_set = fs.FeatureSet(
            name=name,
            entities=[fs.Entity("first_name")],
            timestamp_key="time",
        )

        if partitioned:
            targets = [
                NoSqlTarget(),
                ParquetTarget(
                    name="tar1",
                    path="v3io:///bigdata/sched-t/",
                    partitioned=True,
                    partition_cols=["time"],
                ),
            ]
        else:
            targets = [
                ParquetTarget(
                    name="tar2", path="v3io:///bigdata/sched-f/", partitioned=False
                ),
                NoSqlTarget(),
            ]

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False).apply(mlrun.mount_v3io()),
            targets=targets,
        )
        # ingest starts every round minute.
        sleep(60 - now.second + 10)

        features = [f"{name}.*"]
        vec = fs.FeatureVector("sched_test-vec", features)

        svc = fs.get_online_feature_service(vec)
        try:
            resp = svc.get([{"first_name": "yosi"}, {"first_name": "moshe"}])
            assert resp[0]["data"] == 10
            assert resp[1]["data"] == 2000

            data = pd.DataFrame(
                {
                    "time": [
                        pd.Timestamp("2021-01-10 12:00:00"),
                        pd.Timestamp("2021-01-10 13:00:00"),
                        now + pd.Timedelta(minutes=10),
                        pd.Timestamp("2021-01-09 13:00:00"),
                    ],
                    "first_name": ["moshe", "dina", "katya", "uri"],
                    "data": [50, 10, 25, 30],
                }
            )
            # writing down a remote source
            fs.ingest(data_set, data, targets=[data_target], overwrite=False)

            sleep(60)
            resp = svc.get(
                [
                    {"first_name": "yosi"},
                    {"first_name": "moshe"},
                    {"first_name": "katya"},
                    {"first_name": "dina"},
                    {"first_name": "uri"},
                ]
            )
            assert resp[0]["data"] == 10
            assert resp[1]["data"] == 50
            assert resp[2] is None
            assert resp[3]["data"] == 10
            assert resp[4] is None
        finally:
            svc.close()

        # check offline
        resp = fs.get_offline_features(vec)
        assert len(resp.to_dataframe() == 4)
        assert "uri" not in resp.to_dataframe() and "katya" not in resp.to_dataframe()

    def test_overwrite_single_file(self):
        data = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                    pd.Timestamp("2021-01-10 11:00:00"),
                ],
                "first_name": ["moshe", "yosi"],
                "data": [2000, 10],
            }
        )
        # writing down a remote source
        target2 = ParquetTarget(partitioned=False)
        data_set = fs.FeatureSet("data", entities=[Entity("first_name")])
        fs.ingest(data_set, data, targets=[target2])

        path = data_set.status.targets[0].get_path().get_absolute_path()

        # the job will be scheduled every minute
        cron_trigger = "*/1 * * * *"

        source = ParquetSource("myparquet", schedule=cron_trigger, path=path)

        feature_set = fs.FeatureSet(
            name="overwrite",
            entities=[fs.Entity("first_name")],
            timestamp_key="time",
        )

        targets = [ParquetTarget(path="v3io:///bigdata/bla.parquet", partitioned=False)]

        fs.ingest(
            feature_set,
            source,
            overwrite=True,
            run_config=fs.RunConfig(local=False).apply(mlrun.mount_v3io()),
            targets=targets,
        )
        sleep(60)
        features = ["overwrite.*"]
        vec = fs.FeatureVector("svec", features)

        # check offline
        resp = fs.get_offline_features(vec)
        assert len(resp.to_dataframe()) == 2

    @pytest.mark.parametrize(
        "fixed_window_type",
        [FixedWindowType.CurrentOpenWindow, FixedWindowType.LastClosedWindow],
    )
    def test_query_on_fixed_window(self, fixed_window_type):
        current_time = pd.Timestamp.now()
        data = pd.DataFrame(
            {
                "time": [
                    current_time,
                    current_time - pd.Timedelta(hours=current_time.hour + 2),
                ],
                "first_name": ["moshe", "moshe"],
                "last_name": ["cohen", "cohen"],
                "bid": [2000, 100],
            },
        )
        name = f"measurements_{uuid.uuid4()}"

        # write to kv
        data_set = fs.FeatureSet(
            name, timestamp_key="time", entities=[Entity("first_name")]
        )

        data_set.add_aggregation(
            name="bids",
            column="bid",
            operations=["sum", "max"],
            windows="24h",
        )

        fs.ingest(data_set, data, return_df=True)

        features = [f"{name}.bids_sum_24h", f"{name}.last_name"]

        vector = fs.FeatureVector("my-vec", features)
        with fs.get_online_feature_service(
            vector, fixed_window_type=fixed_window_type
        ) as svc:
            resp = svc.get([{"first_name": "moshe"}])
            if fixed_window_type == FixedWindowType.CurrentOpenWindow:
                expected = {"bids_sum_24h": 2000.0, "last_name": "cohen"}
            else:
                expected = {"bids_sum_24h": 100.0, "last_name": "cohen"}
            assert resp[0] == expected

    def test_split_graph(self):
        quotes_set = fs.FeatureSet("stock-quotes", entities=[fs.Entity("ticker")])

        quotes_set.graph.to("MyMap", "somemap1", field="multi1", multiplier=3).to(
            "storey.Extend", _fn="({'extra': event['bid'] * 77})"
        ).to("storey.Filter", "filter", _fn="(event['bid'] > 70)").to(
            FeaturesetValidator()
        )

        side_step_name = "side-step"
        quotes_set.graph.to(
            "storey.Extend", name=side_step_name, _fn="({'extra2': event['bid'] * 17})"
        )
        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
            fs.preview(quotes_set, quotes)

        non_default_target_name = "side-target"
        quotes_set.set_targets(
            targets=[
                CSVTarget(name=non_default_target_name, after_state=side_step_name)
            ],
            default_final_step="FeaturesetValidator",
        )

        quotes_set.plot(with_targets=True)

        inf_out = fs.preview(quotes_set, quotes)
        ing_out = fs.ingest(quotes_set, quotes, return_df=True)

        default_file_path = quotes_set.get_target_path(TargetTypes.parquet)
        side_file_path = quotes_set.get_target_path(non_default_target_name)

        side_file_out = pd.read_csv(side_file_path)
        default_file_out = pd.read_parquet(default_file_path)
        # default parquet target is partitioned
        default_file_out.drop(
            columns=mlrun.utils.helpers.DEFAULT_TIME_PARTITIONS, inplace=True
        )
        self._split_graph_expected_default.set_index("ticker", inplace=True)

        assert all(self._split_graph_expected_default == default_file_out.round(2))
        assert all(self._split_graph_expected_default == ing_out.round(2))
        assert all(self._split_graph_expected_default == inf_out.round(2))

        assert all(
            self._split_graph_expected_side.sort_index(axis=1)
            == side_file_out.sort_index(axis=1).round(2)
        )

    def test_none_value(self):
        data = pd.DataFrame(
            {"first_name": ["moshe", "yossi"], "bid": [2000, 10], "bool": [True, None]}
        )

        # write to kv
        data_set = fs.FeatureSet("tests2", entities=[Entity("first_name")])
        fs.ingest(data_set, data, return_df=True)
        features = ["tests2.*"]
        vector = fs.FeatureVector("my-vec", features)
        with fs.get_online_feature_service(vector) as svc:
            resp = svc.get([{"first_name": "yossi"}])
            assert resp[0] == {"bid": 10, "bool": None}

    def test_feature_aliases(self):
        df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.038"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                ],
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            }
        )

        # write to kv
        data_set = fs.FeatureSet("aliass", entities=[Entity("ticker")])

        data_set.add_aggregation(
            column="price",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        fs.ingest(data_set, df)
        features = [
            "aliass.price_sum_1h",
            "aliass.price_max_1h as price_m",
        ]
        vector_name = "stocks-vec"
        vector = fs.FeatureVector(vector_name, features)

        resp = fs.get_offline_features(vector).to_dataframe()
        assert len(resp.columns) == 2
        assert "price_m" in resp.columns

        # status should contain original feature name, not its alias
        features_in_status = [feature.name for feature in vector.status.features]
        assert "price_max_1h" in features_in_status
        assert "price_m" not in features_in_status

        vector.save()
        stats = vector.get_stats_table()
        assert len(stats) == 2
        assert "price_m" in stats.index

        svc = fs.get_online_feature_service(vector)
        try:
            resp = svc.get(entity_rows=[{"ticker": "GOOG"}])
            assert resp[0] == {"price_sum_1h": 1441.69, "price_m": 720.92}
        finally:
            svc.close()

        # simulating updating alias from UI
        db = mlrun.get_run_db()
        update_dict = {
            "spec": {
                "features": [
                    "aliass.price_sum_1h as price_s",
                    "aliass.price_max_1h as price_m",
                ]
            }
        }
        db.patch_feature_vector(
            name=vector_name,
            feature_vector_update=update_dict,
            project=self.project_name,
        )

        svc = fs.get_online_feature_service(vector_name)
        try:
            resp = svc.get(entity_rows=[{"ticker": "GOOG"}])
            assert resp[0] == {"price_s": 1441.69, "price_m": 720.92}
        finally:
            svc.close()

        vector = db.get_feature_vector(vector_name, self.project_name, tag="latest")
        stats = vector.get_stats_table()
        assert len(stats) == 2
        assert "price_s" in stats.index

        resp = fs.get_offline_features(vector).to_dataframe()
        assert len(resp.columns) == 2
        assert "price_s" in resp.columns
        assert "price_m" in resp.columns

    def test_forced_columns_target(self):
        columns = ["time", "ask"]
        targets = [ParquetTarget(columns=columns, partitioned=False)]
        quotes_set, _ = prepare_feature_set(
            "forced-columns", "ticker", quotes, timestamp_key="time", targets=targets
        )

        df = pd.read_parquet(quotes_set.get_target_path())
        assert all(df.columns.values == columns)

    def test_csv_parquet_index_alignment(self):
        targets = [CSVTarget()]
        csv_align_set, _ = prepare_feature_set(
            "csv-align", "ticker", quotes, timestamp_key="time", targets=targets
        )
        csv_df = csv_align_set.to_dataframe()

        features = ["csv-align.*"]
        csv_vec = fs.FeatureVector("csv-align-vector", features)
        resp = fs.get_offline_features(csv_vec)
        csv_vec_df = resp.to_dataframe()

        targets = [ParquetTarget(partitioned=False)]
        parquet_align_set, _ = prepare_feature_set(
            "parquet-align", "ticker", quotes, timestamp_key="time", targets=targets
        )
        parquet_df = parquet_align_set.to_dataframe()
        features = ["parquet-align.*"]
        parquet_vec = fs.FeatureVector("parquet-align-vector", features)
        resp = fs.get_offline_features(parquet_vec)
        parquet_vec_df = resp.to_dataframe()

        assert all(csv_df == parquet_df)
        assert all(csv_vec_df == parquet_vec_df)

    @pytest.mark.parametrize("with_columns", [False, True])
    def test_parquet_target_to_dataframe(self, with_columns):
        measurements_partitioned = None
        measurements_nonpartitioned = None
        for partitioned in [False, True]:
            name = f"measurements_{uuid.uuid4()}_{partitioned}"
            key = "patient_id"
            measurements = fs.FeatureSet(
                name, entities=[Entity(key)], timestamp_key="timestamp"
            )
            if partitioned:
                measurements_partitioned = measurements
            else:
                measurements_nonpartitioned = measurements

            source = CSVSource(
                "mycsv",
                path=os.path.relpath(str(self.assets_path / "testdata.csv")),
                time_field="timestamp",
            )

            fs.ingest(
                measurements, source, targets=[ParquetTarget(partitioned=partitioned)]
            )

        columns = ["department", "room"] if with_columns else None
        df_from_partitioned = measurements_partitioned.to_dataframe(columns)
        df_from_nonpartitioned = measurements_nonpartitioned.to_dataframe(columns)
        assert df_from_partitioned.equals(df_from_nonpartitioned)

    def test_sync_pipeline(self):
        stocks_set = fs.FeatureSet(
            "stocks-sync",
            entities=[Entity("ticker", ValueType.STRING)],
            engine="pandas",
        )

        stocks_set.graph.to(name="s1", handler="myfunc1")
        df = fs.ingest(stocks_set, stocks)
        self._logger.info(f"output df:\n{df}")

        features = list(stocks_set.spec.features.keys())
        assert len(features) == 1, "wrong num of features"
        assert "exchange" not in features, "field was not dropped"
        assert len(df) == len(stocks), "dataframe size doesnt match"

    @pytest.mark.parametrize("with_graph", [True, False])
    def test_sync_pipeline_chunks(self, with_graph):
        myset = fs.FeatureSet(
            "early_sense",
            entities=[Entity("patient_id")],
            timestamp_key="timestamp",
            engine="pandas",
        )

        csv_file = os.path.relpath(str(self.assets_path / "testdata.csv"))
        original_df = pd.read_csv(csv_file)
        original_cols = original_df.shape[1]
        print(original_df.shape)
        print(original_df.info())

        chunksize = 100
        source = CSVSource("mycsv", path=csv_file, attributes={"chunksize": chunksize})
        if with_graph:
            myset.graph.to(name="s1", handler="my_func")

        df = fs.ingest(myset, source)
        self._logger.info(f"output df:\n{df}")

        features = list(myset.spec.features.keys())
        print(len(features), features)
        print(myset.to_yaml())
        print(df.shape)
        # original cols - index - timestamp cols
        assert len(features) == original_cols - 2, "wrong num of features"
        assert df.shape[1] == original_cols, "num of cols not as expected"
        # returned DF is only the first chunk (size 100)
        assert df.shape[0] == chunksize, "dataframe size doesnt match"

    def test_target_list_validation(self):
        targets = [ParquetTarget()]
        verify_target_list_fail(targets, with_defaults=True)

        targets = [ParquetTarget(path="path1"), ParquetTarget(path="path2")]
        verify_target_list_fail(targets, with_defaults=False)

        targets = [ParquetTarget(name="parquet1"), ParquetTarget(name="parquet2")]
        verify_target_list_fail(targets)

        targets = [
            ParquetTarget(name="same-name", path="path1"),
            ParquetTarget(name="same-name", path="path2"),
        ]
        verify_target_list_fail(targets, with_defaults=False)

        targets = [
            ParquetTarget(name="parquet1", path="same-path"),
            ParquetTarget(name="parquet2", path="same-path"),
        ]
        verify_target_list_fail(targets)

    def test_same_target_type(self):
        parquet_path1 = str(
            self.results_path / _generate_random_name() / "par1.parquet"
        )
        parquet_path2 = str(
            self.results_path / _generate_random_name() / "par2.parquet"
        )

        targets = [
            ParquetTarget(name="parquet1", path=parquet_path1),
            ParquetTarget(name="parquet2", path=parquet_path2),
        ]
        feature_set, _ = prepare_feature_set(
            "same-target-type", "ticker", quotes, timestamp_key="time", targets=targets
        )
        final_path1 = feature_set.get_target_path(name="parquet1")
        parquet1 = pd.read_parquet(final_path1)
        final_path2 = feature_set.get_target_path(name="parquet2")
        parquet2 = pd.read_parquet(final_path2)

        assert all(parquet1 == quotes.set_index("ticker"))
        assert all(parquet1 == parquet2)

        os.remove(final_path1)
        os.remove(final_path2)

    def test_post_aggregation_step(self):
        quotes_set = fs.FeatureSet("post-aggregation", entities=[fs.Entity("ticker")])
        agg_step = quotes_set.add_aggregation("ask", ["sum", "max"], "1h", "10m")
        agg_step.to("MyMap", "somemap1", field="multi1", multiplier=3)

        # Make sure the map step was added right after the aggregation step
        assert len(quotes_set.graph.states) == 2
        assert quotes_set.graph.states[aggregates_step].after is None
        assert quotes_set.graph.states["somemap1"].after == [aggregates_step]

    def test_featureset_uri(self):
        stocks_set = fs.FeatureSet("stocks01", entities=[fs.Entity("ticker")])
        stocks_set.save()
        fs.ingest(stocks_set.uri, stocks)

    def test_overwrite(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})

        fset = fs.FeatureSet(name="overwrite-fs", entities=[fs.Entity("name")])
        targets = [CSVTarget(), ParquetTarget(partitioned=False), NoSqlTarget()]
        fs.ingest(fset, df1, targets=targets)

        features = ["overwrite-fs.*"]
        fvec = fs.FeatureVector("overwrite-vec", features=features)

        csv_path = fset.get_target_path(name="csv")
        csv_df = pd.read_csv(csv_path)
        assert (
            df1.set_index(keys="name")
            .sort_index()
            .equals(csv_df.set_index(keys="name").sort_index())
        )

        parquet_path = fset.get_target_path(name="parquet")
        parquet_df = pd.read_parquet(parquet_path)
        assert df1.set_index(keys="name").sort_index().equals(parquet_df.sort_index())

        with fs.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "GHI"}])
            assert resp[0]["value"] == 3

        fs.ingest(fset, df2, [ParquetTarget(partitioned=False), NoSqlTarget()])

        csv_path = fset.get_target_path(name="csv")
        csv_df = pd.read_csv(csv_path)
        assert (
            df1.set_index(keys="name")
            .sort_index()
            .equals(csv_df.set_index(keys="name").sort_index())
        )

        parquet_path = fset.get_target_path(name="parquet")
        parquet_df = pd.read_parquet(parquet_path)
        assert df2.set_index(keys="name").sort_index().equals(parquet_df.sort_index())

        with fs.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "GHI"}])
            assert resp[0] is None

            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6

    def test_parquet_target_vector_overwrite(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        fset = fs.FeatureSet(name="fvec-parquet-fset", entities=[fs.Entity("name")])
        fs.ingest(fset, df1)

        features = ["fvec-parquet-fset.*"]
        fvec = fs.FeatureVector("fvec-parquet", features=features)
        fvec.spec.with_indexes = True

        target = ParquetTarget()
        off1 = fs.get_offline_features(fvec, target=target)
        dfout1 = pd.read_parquet(target.get_target_path())

        assert (
            df1.set_index(keys="name")
            .sort_index()
            .equals(off1.to_dataframe().sort_index())
        )
        assert df1.set_index(keys="name").sort_index().equals(dfout1.sort_index())

        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})
        fs.ingest(fset, df2)
        off2 = fs.get_offline_features(fvec, target=target)
        dfout2 = pd.read_parquet(target.get_target_path())
        assert (
            df2.set_index(keys="name")
            .sort_index()
            .equals(off2.to_dataframe().sort_index())
        )
        assert df2.set_index(keys="name").sort_index().equals(dfout2.sort_index())

    def test_overwrite_specified_nosql_path(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})

        targets = [NoSqlTarget(path="v3io:///bigdata/overwrite-spec")]

        fset = fs.FeatureSet(name="overwrite-spec-path", entities=[fs.Entity("name")])
        features = ["overwrite-spec-path.*"]
        fvec = fs.FeatureVector("overwrite-spec-path-fvec", features=features)

        fs.ingest(fset, df1, targets=targets)

        fs.ingest(fset, df2, targets=targets)

        with fs.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6
            resp = svc.get(entity_rows=[{"name": "ABC"}])
            assert resp[0] is None

    def test_overwrite_single_parquet_file(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})

        targets = [ParquetTarget(path="v3io:///bigdata/overwrite-pq-spec/my.parquet")]

        fset = fs.FeatureSet(
            name="overwrite-pq-spec-path", entities=[fs.Entity("name")]
        )

        fs.ingest(fset, df1, targets=targets)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(fset, df2, targets=targets, overwrite=False)

    def test_overwrite_false(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})
        df3 = pd.concat([df1, df2])

        fset = fs.FeatureSet(name="override-false", entities=[fs.Entity("name")])
        fs.ingest(fset, df1)

        features = ["override-false.*"]
        fvec = fs.FeatureVector("override-false-vec", features=features)
        fvec.spec.with_indexes = True

        off1 = fs.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        fs.ingest(fset, df2, overwrite=False)

        off2 = fs.get_offline_features(fvec).to_dataframe()
        assert df3.set_index(keys="name").sort_index().equals(off2.sort_index())

        fs.ingest(fset, df1, targets=[ParquetTarget()])

        off1 = fs.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        with fs.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(fset, df1, targets=[CSVTarget()], overwrite=False)

        fset.set_targets(targets=[CSVTarget()])
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(fset, df1, overwrite=False)

    def test_purge_v3io(self):
        key = "patient_id"
        fset = fs.FeatureSet("purge", entities=[Entity(key)], timestamp_key="timestamp")
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
            time_field="timestamp",
        )
        targets = [
            CSVTarget(),
            CSVTarget(name="specified-path", path="v3io:///bigdata/csv-purge-test.csv"),
            ParquetTarget(partitioned=True, partition_cols=["timestamp"]),
            NoSqlTarget(),
        ]
        fset.set_targets(
            targets=targets,
            with_defaults=False,
        )
        fs.ingest(fset, source)

        verify_purge(fset, targets)

        fs.ingest(fset, source)

        targets_to_purge = targets[:-1]

        verify_purge(fset, targets_to_purge)

    def test_purge_redis(self):
        key = "patient_id"
        fset = fs.FeatureSet("purge", entities=[Entity(key)], timestamp_key="timestamp")
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
            time_field="timestamp",
        )
        targets = [
            CSVTarget(),
            CSVTarget(name="specified-path", path="v3io:///bigdata/csv-purge-test.csv"),
            ParquetTarget(partitioned=True, partition_cols=["timestamp"]),
            RedisNoSqlTarget(),
        ]
        fset.set_targets(
            targets=targets,
            with_defaults=False,
        )
        fs.ingest(fset, source)

        verify_purge(fset, targets)

        fs.ingest(fset, source)

        targets_to_purge = targets[:-1]

        verify_purge(fset, targets_to_purge)

    # After moving to run on a new system test environment this test was running for 75 min and then failing
    # skipping until it get fixed as this results all the suite to run much longer
    @pytest.mark.timeout(180)
    def test_purge_nosql(self):
        def get_v3io_api_host():
            """Return only the host out of v3io_api

            Takes the parameter from config and strip it from it's protocol and port
            returning only the host name.
            """
            api = None
            if config.v3io_api:
                api = config.v3io_api
                if "//" in api:
                    api = api[api.find("//") + 2 :]
                if ":" in api:
                    api = api[: api.find(":")]
            return api

        key = "patient_id"
        fset = fs.FeatureSet(
            name="nosqlpurge", entities=[Entity(key)], timestamp_key="timestamp"
        )
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
            time_field="timestamp",
        )
        targets = [
            NoSqlTarget(
                name="nosql", path="v3io:///bigdata/system-test-project/nosql-purge"
            ),
            NoSqlTarget(
                name="fullpath",
                path=f"v3io://webapi.{get_v3io_api_host()}/bigdata/system-test-project/nosql-purge-full",
            ),
        ]

        for tar in targets:
            test_target = [tar]
            fset.set_targets(
                with_defaults=False,
                targets=test_target,
            )
            self._logger.info(f"ingesting with target {tar.name}")
            fs.ingest(fset, source)
            self._logger.info(f"purging target {tar.name}")
            verify_purge(fset, test_target)

    def test_ingest_dataframe_index(self):
        orig_df = pd.DataFrame([{"x", "y"}])
        orig_df.index.name = "idx"

        fset = fs.FeatureSet("myfset", entities=[Entity("idx")])
        fs.ingest(
            fset, orig_df, [ParquetTarget()], infer_options=fs.InferOptions.default()
        )

    def test_ingest_with_column_conversion(self):
        orig_df = source = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2002-04-01 04:32:34.000"),
                    pd.Timestamp("2002-04-01 15:05:37.000"),
                    pd.Timestamp("2002-03-31 23:46:07.000"),
                ],
                "ssrxbtok": [488441267876, 438975336749, 298802679370],
                "nkxuonfx": [0.241233, 0.160264, 0.045345],
                "xzvipbmo": [True, False, None],
                "bikyseca": ["ONE", "TWO", "THREE"],
                "napxsuhp": [True, False, True],
                "oegndrxe": [
                    pd.Timestamp("2002-04-01 04:32:34.000"),
                    pd.Timestamp("2002-04-01 05:06:34.000"),
                    pd.Timestamp("2002-04-01 05:38:34.000"),
                ],
                "aatxnkgx": [-227504700006, -470002151801, -33193685176],
                "quupyoxi": ["FOUR", "FIVE", "SIX"],
                "temdojgz": [0.570031, 0.677182, 0.276053],
            },
            index=None,
        )

        fset = fs.FeatureSet(
            "rWQTKqbhje",
            timestamp_key="time_stamp",
            entities=[
                Entity("{}".format(k["name"]))
                for k in [
                    {
                        "dtype": "float",
                        "null_values": False,
                        "name": "temdojgz",
                        "df_dtype": "float64",
                    },
                    {
                        "dtype": "str",
                        "null_values": False,
                        "name": "bikyseca",
                        "df_dtype": "object",
                    },
                    {
                        "dtype": "float",
                        "null_values": False,
                        "name": "nkxuonfx",
                        "df_dtype": "float64",
                    },
                ]
            ],
        )

        fset.graph.to(name="s1", handler="my_func")
        ikjqkfcz = ParquetTarget(path="v3io:///bigdata/ifrlsjvxgv", partitioned=False)
        fs.ingest(fset, source, targets=[ikjqkfcz])

        features = ["rWQTKqbhje.*"]
        vector = fs.FeatureVector("WPAyrYux", features)
        vector.spec.with_indexes = True
        resp = fs.get_offline_features(vector)
        off_df = resp.to_dataframe()
        if None in list(orig_df.index.names):
            orig_df.set_index(["temdojgz", "bikyseca", "nkxuonfx"], inplace=True)
        orig_df = orig_df.sort_values(
            by=["temdojgz", "bikyseca", "nkxuonfx"]
        ).sort_index(axis=1)
        off_df = off_df.sort_values(by=["temdojgz", "bikyseca", "nkxuonfx"]).sort_index(
            axis=1
        )
        pd.testing.assert_frame_equal(
            off_df,
            orig_df,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_like=True,
            check_names=True,
        )

    def test_stream_source(self):
        # create feature set, ingest sample data and deploy nuclio function with stream source
        fset_name = "a2-stream_test"
        myset = FeatureSet(f"{fset_name}", entities=[Entity("ticker")])
        fs.ingest(myset, quotes)
        source = StreamSource(key_field="ticker", time_field="time")
        filename = str(
            pathlib.Path(tests.conftest.tests_root_directory)
            / "api"
            / "runtimes"
            / "assets"
            / "sample_function.py"
        )

        function = mlrun.code_to_function(
            "ingest_transactions", kind="serving", filename=filename
        )
        function.spec.default_content_type = "application/json"
        run_config = fs.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fs.deploy_ingestion_service(
            featureset=myset, source=source, run_config=run_config
        )
        # push records to stream
        stream_path = f"v3io:///projects/{function.metadata.project}/FeatureStore/{fset_name}/v3ioStream"
        events_pusher = mlrun.datastore.get_stream_pusher(stream_path)
        client = mlrun.platforms.V3ioStreamClient(stream_path, seek_to="EARLIEST")
        events_pusher.push(
            {
                "ticker": "AAPL",
                "time": "2021-08-15T10:58:37.415101",
                "bid": 300,
                "ask": 100,
            }
        )
        # verify new records in stream
        resp = client.get_records()
        assert len(resp) != 0
        # read from online service updated data

        vector = fs.FeatureVector("my-vec", [f"{fset_name}.*"])
        with fs.get_online_feature_service(vector) as svc:
            sleep(5)
            resp = svc.get([{"ticker": "AAPL"}])

        assert resp[0]["bid"] == 300

    def test_get_offline_from_feature_set_with_no_schema(self):
        myset = FeatureSet("fset2", entities=[Entity("ticker")])
        fs.ingest(myset, quotes, infer_options=InferOptions.Null)
        features = ["fset2.*"]
        vector = fs.FeatureVector("QVMytLdP", features, with_indexes=True)

        try:
            fs.get_offline_features(vector)
            assert False
        except mlrun.errors.MLRunInvalidArgumentError:
            pass

    def test_join_with_table(self):
        table_url = "v3io:///bigdata/system-test-project/nosql/test_join_with_table"

        df = pd.DataFrame({"name": ["ABC", "DEF"], "aug": ["1", "2"]})
        fset = fs.FeatureSet(
            name="test_join_with_table_fset", entities=[fs.Entity("name")]
        )
        fs.ingest(fset, df, targets=[NoSqlTarget(path=table_url)])
        run_id = fset.status.targets[0].run_id
        table_url = f"{table_url}/{run_id}"
        df = pd.DataFrame(
            {
                "key": ["mykey1", "mykey2", "mykey3"],
                "foreignkey1": ["AB", "DE", "GH"],
                "foreignkey2": ["C", "F", "I"],
            }
        )

        fset = fs.FeatureSet("myfset", entities=[Entity("key")])
        fset.set_targets([], with_defaults=False)
        fset.graph.to(
            "storey.JoinWithTable",
            table=table_url,
            _key_extractor="(event['foreignkey1'] + event['foreignkey2'])",
            attributes=["aug"],
            inner_join=True,
        )
        df = fs.ingest(fset, df, targets=[], infer_options=fs.InferOptions.default())
        assert df.to_dict() == {
            "foreignkey1": {"mykey1": "AB", "mykey2": "DE"},
            "foreignkey2": {"mykey1": "C", "mykey2": "F"},
            "aug": {"mykey1": "1", "mykey2": "2"},
        }

    def test_get_offline_features_with_tag(self):
        def validate_result(test_vector, test_keys):
            res_set = fs.get_offline_features(test_vector)
            assert res_set is not None
            res_keys = list(res_set.vector.status.stats.keys())
            assert res_keys.sort() == test_keys.sort()

        data = quotes
        name = "quotes"
        tag = "test"
        project = self.project_name

        test_set = fs.FeatureSet(name, entities=[Entity("ticker", ValueType.STRING)])

        df = fs.ingest(test_set, data)
        assert df is not None

        # change feature set and save with tag
        test_set.add_aggregation(
            "bid",
            ["avg"],
            "1h",
        )
        new_column = "bid_avg_1h"
        test_set.metadata.tag = tag
        fs.ingest(test_set, data)

        # retrieve feature set with feature vector and check for changes
        vector = fs.FeatureVector("vector", [f"{name}.*"], with_indexes=True)
        vector_with_tag = fs.FeatureVector(
            "vector_with_tag", [f"{name}:{tag}.*"], with_indexes=True
        )
        vector_with_project = fs.FeatureVector(
            "vector_with_project", [f"{project}/{name}.*"], with_indexes=True
        )
        # vector_with_project.metadata.project = "bs"
        vector_with_features = fs.FeatureVector(
            "vector_with_features", [f"{name}.bid", f"{name}.time"], with_indexes=True
        )
        vector_with_project_tag_and_features = fs.FeatureVector(
            "vector_with_project_tag_and_features",
            [f"{project}/{name}:{tag}.bid", f"{project}/{name}:{tag}.{new_column}"],
            with_indexes=True,
        )

        expected_keys = ["time", "bid", "ask"]

        for vec, keys in [
            (vector, expected_keys),
            (vector_with_tag, expected_keys + [new_column]),
            (vector_with_project, expected_keys),
            (vector_with_features, ["bid", "time"]),
            (vector_with_project_tag_and_features, ["bid", new_column]),
        ]:
            validate_result(vec, keys)

    def test_get_online_feature_service_with_tag(self):
        def validate_result(test_vector, test_keys):
            with fs.get_online_feature_service(test_vector) as svc:
                sleep(5)
                resp = svc.get([{"ticker": "AAPL"}])
            assert resp is not None
            resp_keys = list(resp[0].keys())
            assert resp_keys.sort() == test_keys.sort()

        data = quotes
        name = "quotes"
        tag = "test"
        project = self.project_name

        test_set = fs.FeatureSet(name, entities=[Entity("ticker", ValueType.STRING)])

        df = fs.ingest(test_set, data)
        assert df is not None

        # change feature set and save with tag
        test_set.add_aggregation(
            "bid",
            ["avg"],
            "1h",
        )
        new_column = "bid_avg_1h"
        test_set.metadata.tag = tag
        fs.ingest(test_set, data)

        # retrieve feature set with feature vector and check for changes
        vector = fs.FeatureVector("vector", [f"{name}.*"], with_indexes=True)
        vector_with_tag = fs.FeatureVector(
            "vector_with_tag", [f"{name}:{tag}.*"], with_indexes=True
        )
        vector_with_project = fs.FeatureVector(
            "vector_with_project", [f"{project}/{name}.*"], with_indexes=True
        )
        # vector_with_project.metadata.project = "bs"
        vector_with_features = fs.FeatureVector(
            "vector_with_features", [f"{name}.bid", f"{name}.time"], with_indexes=True
        )
        vector_with_project_tag_and_features = fs.FeatureVector(
            "vector_with_project_tag_and_features",
            [f"{project}/{name}:{tag}.bid", f"{project}/{name}:{tag}.{new_column}"],
            with_indexes=True,
        )

        expected_keys = ["ticker", "time", "bid", "ask"]

        for vec, keys in [
            (vector, expected_keys),
            (vector_with_tag, expected_keys + [new_column]),
            (vector_with_project, expected_keys),
            (vector_with_features, ["bid", "time"]),
            (vector_with_project_tag_and_features, ["bid", new_column]),
        ]:
            validate_result(vec, keys)

    def test_preview_saves_changes(self):
        name = "update-on-preview"
        v3io_source = StreamSource(key_field="ticker", time_field="time")
        fset = fs.FeatureSet(name, timestamp_key="time", entities=[Entity("ticker")])
        import v3io.dataplane

        v3io_client = v3io.dataplane.Client()

        stream_path = f"/{self.project_name}/FeatureStore/{name}/v3ioStream"
        try:
            v3io_client.stream.delete(
                container="projects",
                stream_path=stream_path,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
        except RuntimeError as err:
            assert err.__str__().__contains__(
                "404"
            ), "only acceptable error is with status 404"
        finally:
            v3io_client.stream.create(
                container="projects", stream_path=stream_path, shard_count=1
            )

        record = {
            "data": json.dumps(
                {
                    "ticker": "AAPL",
                    "time": "2021-08-15T10:58:37.415101",
                    "bid": 300,
                    "ask": 100,
                }
            )
        }

        v3io_client.stream.put_records(
            container="projects", stream_path=stream_path, records=[record]
        )

        fs.preview(
            featureset=fset,
            source=quotes,
            entity_columns=["ticker"],
            timestamp_key="time",
        )

        filename = str(
            pathlib.Path(tests.conftest.tests_root_directory)
            / "api"
            / "runtimes"
            / "assets"
            / "sample_function.py"
        )

        function = mlrun.code_to_function(
            "ingest_transactions", kind="serving", filename=filename
        )
        function.spec.default_content_type = "application/json"
        run_config = fs.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fs.deploy_ingestion_service(
            featureset=fset,
            source=v3io_source,
            run_config=run_config,
            targets=[ParquetTarget(flush_after_seconds=1)],
        )

        record = {
            "data": json.dumps(
                {
                    "ticker": "AAPL",
                    "time": "2021-08-15T10:58:37.415101",
                    "bid": 400,
                    "ask": 200,
                }
            )
        }

        v3io_client.stream.put_records(
            container="projects", stream_path=stream_path, records=[record]
        )

        features = [f"{name}.*"]
        vector = fs.FeatureVector("vecc", features, with_indexes=True)

        fs.get_offline_features(vector)

    def test_online_impute(self):
        data = pd.DataFrame(
            {
                "time_stamp": [
                    pd.Timestamp("2016-05-25 13:31:00.000"),
                    pd.Timestamp("2016-05-25 13:32:00.000"),
                    pd.Timestamp("2016-05-25 13:33:00.000"),
                ],
                "data": [10, 20, 60],
                "name": ["ab", "cd", "ef"],
            }
        )

        data_set1 = fs.FeatureSet(
            "imp1", entities=[Entity("name")], timestamp_key="time_stamp"
        )
        data_set1.add_aggregation(
            "data",
            ["avg", "max"],
            "1h",
        )
        fs.ingest(data_set1, data, infer_options=fs.InferOptions.default())

        data2 = pd.DataFrame({"data2": [1, None, np.inf], "name": ["ab", "cd", "ef"]})

        data_set2 = fs.FeatureSet("imp2", entities=[Entity("name")])
        fs.ingest(data_set2, data2, infer_options=fs.InferOptions.default())

        features = ["imp2.data2", "imp1.data_max_1h", "imp1.data_avg_1h"]

        # create vector and online service with imputing policy
        vector = fs.FeatureVector("vectori", features)
        with fs.get_online_feature_service(
            vector, impute_policy={"*": "$max", "data_avg_1h": "$mean", "data2": 4}
        ) as svc:
            print(svc.vector.status.to_yaml())

            resp = svc.get([{"name": "ab"}])
            assert resp[0]["data2"] == 1
            assert resp[0]["data_max_1h"] == 60
            assert resp[0]["data_avg_1h"] == 30

            # test as list
            resp = svc.get([{"name": "ab"}], as_list=True)
            assert resp == [[1, 60, 30]]

            # test with missing key
            resp = svc.get([{"name": "xx"}])
            assert resp == [None]

            # test with missing key, as list
            resp = svc.get([{"name": "xx"}], as_list=True)
            assert resp == [None]

            resp = svc.get([{"name": "cd"}])
            assert resp[0]["data2"] == 4
            assert resp[0]["data_max_1h"] == 60
            assert resp[0]["data_avg_1h"] == 30

            resp = svc.get([{"name": "ef"}])
            assert resp[0]["data2"] == 4
            assert resp[0]["data_max_1h"] == 60
            assert resp[0]["data_avg_1h"] == 30

        # check without impute
        vector = fs.FeatureVector("vectori2", features)
        with fs.get_online_feature_service(vector) as svc:
            resp = svc.get([{"name": "cd"}])
            assert np.isnan(resp[0]["data2"])
            assert np.isnan(resp[0]["data_avg_1h"])

    def test_map_with_state_with_table(self):
        table_url = (
            "v3io:///bigdata/system-test-project/nosql/test_map_with_state_with_table"
        )

        df = pd.DataFrame({"name": ["a", "b"], "sum": [11, 22]})
        fset = fs.FeatureSet(
            name="test_map_with_state_with_table_fset", entities=[fs.Entity("name")]
        )
        fs.ingest(fset, df, targets=[NoSqlTarget(path=table_url)])
        table_url_with_run_uid = fset.status.targets[0].get_path().get_absolute_path()
        df = pd.DataFrame({"key": ["a", "a", "b"], "x": [2, 3, 4]})

        fset = fs.FeatureSet("myfset", entities=[Entity("key")])
        fset.set_targets([], with_defaults=False)
        fset.graph.to(
            "storey.MapWithState",
            initial_state=table_url_with_run_uid,
            group_by_key=True,
            _fn="map_with_state_test_function",
        )
        df = fs.ingest(fset, df, targets=[], infer_options=fs.InferOptions.default())
        assert df.to_dict() == {
            "name": {"a": "a", "b": "b"},
            "sum": {"a": 16, "b": 26},
        }

    def test_get_online_feature_service(self):
        vector = self._generate_vector()
        with fs.get_online_feature_service(vector) as svc:
            resp = svc.get([{"name": "ab"}])
            assert resp[0] == {"data": 10}

    def test_allow_empty_vector(self):
        # test that we can pass an non materialized vector to function using special flag
        vector = fs.FeatureVector("dummy-vec", [])
        vector.save()

        func = mlrun.new_function("myfunc", kind="job", handler="myfunc").with_code(
            body=myfunc
        )
        func.spec.allow_empty_resources = True
        run = func.run(inputs={"data": vector.uri}, local=True)
        assert run.output("uri") == vector.uri

    def test_two_ingests(self):
        df1 = pd.DataFrame({"name": ["AB", "CD"], "some_data": [10, 20]})
        set1 = fs.FeatureSet("set1", entities=[Entity("name")])
        fs.ingest(set1, df1)

        df2 = pd.DataFrame({"name": ["AB", "CD"], "some_data": ["Paris", "Tel Aviv"]})
        set2 = fs.FeatureSet("set2", entities=[Entity("name")])
        fs.ingest(set2, df2)
        vector = fs.FeatureVector("check", ["set1.*", "set2.some_data as ddata"])
        svc = fs.get_online_feature_service(vector)

        try:
            resp = svc.get([{"name": "AB"}])
        finally:
            svc.close()
        assert resp == [{"some_data": 10, "ddata": "Paris"}]

        resp = fs.get_offline_features(vector)
        assert resp.to_dataframe().to_dict() == {
            "some_data": {0: 10, 1: 20},
            "ddata": {0: "Paris", 1: "Tel Aviv"},
        }

    @pytest.mark.parametrize(
        "targets, feature_set_targets, expected_target_names",
        [
            [None, None, ["parquet", "nosql"]],
            [[ParquetTarget("par")], None, ["par"]],
            [[ParquetTarget("par", "v3io:///bigdata/dis-dt.parquet")], None, ["par"]],
            [None, [ParquetTarget("par2")], ["par2"]],
            [[ParquetTarget("par")], [ParquetTarget("par2")], ["par"]],
        ],
    )
    def test_deploy_ingestion_service_with_different_targets(
        self, targets, feature_set_targets, expected_target_names
    ):
        fset_name = "dis-set"
        fset = FeatureSet(f"{fset_name}", entities=[Entity("ticker")])

        if feature_set_targets:
            fset.set_targets(feature_set_targets, with_defaults=False)
        fs.ingest(fset, quotes)
        source = StreamSource(key_field="ticker", time_field="time")
        filename = str(
            pathlib.Path(tests.conftest.tests_root_directory)
            / "api"
            / "runtimes"
            / "assets"
            / "sample_function.py"
        )

        function = mlrun.code_to_function(
            "ingest_transactions", kind="serving", filename=filename
        )
        function.spec.default_content_type = "application/json"
        function.spec.image_pull_policy = "Always"
        run_config = fs.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fs.deploy_ingestion_service(
            featureset=fset, source=source, run_config=run_config, targets=targets
        )

        fset.reload()  # refresh to ingestion service updates
        assert fset.status.targets is not None
        for actual_tar in fset.status.targets:
            assert actual_tar.run_id is not None
        for expected in expected_target_names:
            assert fset.get_target_path(expected) is not None

    def test_feature_vector_with_all_features_and_label_feature(self):
        feature_set = FeatureSet("fs-label", entities=[Entity("ticker")])
        fs.ingest(feature_set, stocks)
        expected = stocks.to_dict()
        expected.pop("ticker")

        fv = fs.FeatureVector("fv-label", ["fs-label.*"], "fs-label.name")
        res = fs.get_offline_features(fv)

        assert res is not None
        assert res.to_dataframe().to_dict() == expected

    def test_get_offline_for_two_feature_set_with_same_column_name(self):
        # This test is testing that all columns are returned with no failure even though
        # two features sets and the label column has the column 'name'.
        expected = ["fs1_exchange", "name", "fs2_name", "fs2_exchange"]

        feature_set = FeatureSet("fs1", entities=[Entity("ticker")])
        fs.ingest(feature_set, stocks)
        feature_set = FeatureSet("fs2", entities=[Entity("ticker")])
        fs.ingest(feature_set, stocks)

        fv = fs.FeatureVector("fv-label", ["fs1.* as fs1", "fs2.* as fs2"], "fs1.name")
        res = fs.get_offline_features(fv)

        assert res is not None
        assert len(expected) == len(res.to_dataframe().to_dict().keys())
        for key in res.to_dataframe().to_dict().keys():
            assert key in expected

    @pytest.mark.parametrize("engine", ["local", "dask"])
    def test_get_offline_features_with_filter(self, engine):
        engine_args = {}
        if engine == "dask":
            dask_cluster = mlrun.new_function(
                "dask_tests", kind="dask", image="mlrun/ml-models"
            )
            dask_cluster.apply(mlrun.mount_v3io())
            dask_cluster.spec.remote = True
            dask_cluster.with_requests(mem="2G")
            dask_cluster.save()
            engine_args = {
                "dask_client": dask_cluster,
                "dask_cluster_uri": dask_cluster.uri,
            }

        data = pd.DataFrame(
            {
                "name": ["A", "B", "C", "D", "E"],
                "age": [33, 4, 76, 90, 24],
                "department": ["IT", "RD", "RD", "Marketing", "IT"],
            },
            index=[0, 1, 2, 3, 4],
        )
        data["id"] = data.index

        one_hot_encoder_mapping = {
            "department": list(data["department"].unique()),
        }
        data_set = FeatureSet(
            "fs-new", entities=[Entity("id")], description="feature set"
        )
        data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        data_set.set_targets()
        fs.ingest(data_set, data, infer_options=fs.InferOptions.default())

        fv_name = "new-fv"
        features = [
            "fs-new.name",
            "fs-new.age",
            "fs-new.department_RD",
            "fs-new.department_IT",
            "fs-new.department_Marketing",
        ]

        my_fv = fs.FeatureVector(fv_name, features, description="my feature vector")
        my_fv.save()
        # expected data frame
        expected_df = pd.DataFrame(
            {
                "name": ["C"],
                "age": [76],
                "department_RD": [1],
                "department_IT": [0],
                "department_Marketing": [0],
            },
            index=[0],
        )

        # different tests
        result_1 = fs.get_offline_features(
            fv_name,
            target=ParquetTarget(),
            query="age>6 and department_RD==1",
            engine=engine,
            engine_args=engine_args,
        )
        df_res_1 = result_1.to_dataframe()

        assert df_res_1.equals(expected_df)

        result_2 = fs.get_offline_features(
            fv_name,
            target=ParquetTarget(),
            query="name in ['C']",
            engine=engine,
            engine_args=engine_args,
        )
        df_res_2 = result_2.to_dataframe()

        assert df_res_2.equals(expected_df)

    def test_set_event_with_spaces_or_hyphens(self):

        lst_1 = [
            " Private",
            " Private",
            " Local-gov",
            " Private",
        ]
        lst_2 = [0, 1, 2, 3]
        lst_3 = [25, 38, 28, 44]
        data = pd.DataFrame(
            list(zip(lst_2, lst_1, lst_3)), columns=["id", "workclass", "age"]
        )
        # One Hot Encode the newly defined mappings
        one_hot_encoder_mapping = {"workclass": list(data["workclass"].unique())}

        # Define the corresponding FeatureSet
        data_set = FeatureSet(
            "test", entities=[Entity("id")], description="feature set"
        )

        data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        data_set.set_targets()

        df_res = fs.ingest(data_set, data, infer_options=fs.InferOptions.default())

        expected_df = pd.DataFrame(
            list(zip([1, 1, 0, 1], [0, 0, 1, 0], lst_3)),
            columns=["workclass__Private", "workclass__Local_gov", "age"],
        )

        assert df_res.equals(expected_df)

    def test_onehot_with_int_values(self):
        lst_1 = [0, 0, 1, 0]
        lst_2 = [0, 1, 2, 3]
        lst_3 = [25, 38, 28, 44]
        data = pd.DataFrame(
            list(zip(lst_2, lst_1, lst_3)), columns=["id", "workclass", "age"]
        )
        # One Hot Encode the newly defined mappings
        one_hot_encoder_mapping = {"workclass": list(data["workclass"].unique())}

        # Define the corresponding FeatureSet
        data_set = FeatureSet(
            "test", entities=[Entity("id")], description="feature set"
        )

        data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        data_set.set_targets()

        df_res = fs.ingest(data_set, data, infer_options=fs.InferOptions.default())

        expected_df = pd.DataFrame(
            list(zip([1, 1, 0, 1], [0, 0, 1, 0], lst_3)),
            columns=["workclass_0", "workclass_1", "age"],
        )

        assert df_res.equals(expected_df)

    def test_onehot_with_array_values(self):
        lst_1 = [[1, 2], [1, 2], [0, 1], [1, 2]]
        lst_2 = [0, 1, 2, 3]
        lst_3 = [25, 38, 28, 44]
        data = pd.DataFrame(
            list(zip(lst_2, lst_1, lst_3)), columns=["id", "workclass", "age"]
        )
        # One Hot Encode the newly defined mappings
        one_hot_encoder_mapping = {"workclass": [[1, 2], [0, 1]]}

        # Define the corresponding FeatureSet
        data_set = FeatureSet(
            "test", entities=[Entity("id")], description="feature set"
        )
        with pytest.raises(ValueError):
            data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
            data_set.set_targets()
            fs.ingest(data_set, data, infer_options=fs.InferOptions.default())

    @pytest.mark.skipif(not kafka_brokers, reason="KAFKA_BROKERS must be set")
    def test_kafka_target(self, kafka_consumer):

        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fs.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        target = KafkaTarget(
            "kafka",
            path=kafka_topic,
            bootstrap_servers=kafka_brokers,
        )
        fs.ingest(stocks_set, stocks, [target])

        expected_records = [
            b'{"ticker": "MSFT", "name": "Microsoft Corporation", "booly": true}',
            b'{"ticker": "GOOG", "name": "Alphabet Inc", "booly": false}',
            b'{"ticker": "AAPL", "name": "Apple Inc", "booly": true}',
        ]

        kafka_consumer.subscribe([kafka_topic])
        for expected_record in expected_records:
            record = next(kafka_consumer)
            assert record.value == expected_record

    @pytest.mark.skipif(kafka_brokers == "", reason="KAFKA_BROKERS must be set")
    def test_kafka_target_bad_kafka_options(self):

        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fs.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        target = KafkaTarget(
            "kafka",
            path=kafka_topic,
            bootstrap_servers=kafka_brokers,
            producer_options={"compression_type": "invalid value"},
        )
        with pytest.raises(ValueError):
            fs.ingest(stocks_set, stocks, [target])

    def test_alias_change(self):
        quotes = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.030"),
                    pd.Timestamp("2016-05-25 13:30:00.041"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.049"),
                    pd.Timestamp("2016-05-25 13:30:00.072"),
                    pd.Timestamp("2016-05-25 13:30:00.075"),
                ],
                "ticker": [
                    "GOOG",
                    "MSFT",
                    "MSFT",
                    "MSFT",
                    "GOOG",
                    "AAPL",
                    "GOOG",
                    "MSFT",
                ],
                "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
                "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
            }
        )

        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            }
        )

        stocks_set = fs.FeatureSet("stocks", entities=[fs.Entity("ticker")])
        fs.ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())

        quotes_set = fs.FeatureSet("stock-quotes", entities=[fs.Entity("ticker")])

        quotes_set.graph.to("storey.Extend", _fn="({'extra': event['bid'] * 77})").to(
            "storey.Filter", "filter", _fn="(event['bid'] > 51.92)"
        ).to(FeaturesetValidator())

        quotes_set.add_aggregation("asks1", ["sum", "max"], "1h", "10m")
        quotes_set.add_aggregation("asks5", ["sum", "max"], "5h", "10m")
        quotes_set.add_aggregation("bids", ["min", "max"], "1h", "10m")

        quotes_set["bid"] = fs.Feature(
            validator=MinMaxValidator(min=52, severity="info")
        )

        quotes_set.set_targets()

        fs.preview(
            quotes_set,
            quotes,
            entity_columns=["ticker"],
            timestamp_key="time",
            options=fs.InferOptions.default(),
        )

        fs.ingest(quotes_set, quotes)

        features = [
            "stock-quotes.asks5_sum_5h as total_ask",
            "stock-quotes.bids_min_1h",
            "stock-quotes.bids_max_1h",
            "stocks.*",
        ]

        vector_name = "stocks-vec"

        vector = fs.FeatureVector(
            vector_name, features, description="stocks demo feature vector"
        )
        vector.save()

        # change alias
        request_url = (
            f"{mlrun.mlconf.iguazio_api_url}/mlrun/api/v1/projects/{self.project_name}/"
            f"feature-vectors/{vector_name}/references/latest"
        )
        request_body = {
            "metadata": {},
            "spec": {
                "features": [
                    "stock-quotes.asks5_sum_5h as new_alias_for_total_ask",
                    "stock-quotes.bids_min_1h",
                    "stock-quotes.bids_max_1h",
                    "stocks.*",
                ]
            },
        }
        headers = {
            "Cookie": "session=j:" + json.dumps({"sid": os.getenv("V3IO_ACCESS_KEY")})
        }
        response = requests.patch(
            request_url, json=request_body, headers=headers, verify=False
        )
        assert (
            response.status_code == 200
        ), f"Failed to patch feature vector: {response}"

        service = fs.get_online_feature_service(vector_name)
        try:
            resp = service.get([{"ticker": "AAPL"}])
            assert resp == [
                {
                    "new_alias_for_total_ask": 0.0,
                    "bids_min_1h": math.inf,
                    "bids_max_1h": -math.inf,
                    "name": "Apple Inc",
                    "exchange": "NASDAQ",
                }
            ]
            resp = service.get([{"ticker": "AAPL"}], as_list=True)
            assert resp == [[0.0, math.inf, -math.inf, "Apple Inc", "NASDAQ"]]
        finally:
            service.close()

    @pytest.mark.parametrize("engine", ["local", "dask"])
    def test_relation_join(self, engine):
        """Test 3 option of using get offline feature with relations"""
        departments = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "name": [
                    "dept1",
                    "dept2",
                    "dept3",
                    "dept4",
                    "dept5",
                    "dept6",
                    "dept7",
                    "dept8",
                    "dept9",
                    "dept10",
                ],
            }
        )

        employees = pd.DataFrame(
            {
                "id": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "name": [
                    "name1",
                    "name2",
                    "name3",
                    "name4",
                    "name5",
                    "name6",
                    "name7",
                    "name8",
                    "name9",
                    "name10",
                ],
                "department_id": [1, 1, 1, 2, 2, 2, 6, 6, 6, 11],
            }
        )

        result = pd.DataFrame(
            {
                "n": [
                    "name1",
                    "name2",
                    "name3",
                    "name4",
                    "name5",
                    "name6",
                    "name7",
                    "name8",
                    "name9",
                ],
                "n2": [
                    "dept1",
                    "dept1",
                    "dept1",
                    "dept2",
                    "dept2",
                    "dept2",
                    "dept6",
                    "dept6",
                    "dept6",
                ],
            }
        )

        engine_args = {}
        if engine == "dask":
            dask_cluster = mlrun.new_function(
                "dask_tests", kind="dask", image="mlrun/ml-models"
            )
            dask_cluster.apply(mlrun.mount_v3io())
            dask_cluster.spec.remote = True
            dask_cluster.with_requests(mem="2G")
            dask_cluster.save()
            engine_args = {
                "dask_client": dask_cluster,
                "dask_cluster_uri": dask_cluster.uri,
            }
        # relations according to departments_set relations
        departments_set = fs.FeatureSet(
            "departments",
            entities=[fs.Entity("id")],
            relations={"employees": {"id": "department_id"}},
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(departments_set, departments)
        employees_set = fs.FeatureSet(
            "employees",
            entities=[fs.Entity("id")],
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(employees_set, employees)

        features = ["employees.name as n", "departments.name as n2"]

        vector = fs.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp_1 = fs.get_offline_features(
            vector,
            join_type="inner",
            engine_args=engine_args,
        )
        assert_frame_equal(result, resp_1.to_dataframe())

        # relations according to employees_set relations
        departments_set = fs.FeatureSet(
            "departments",
            entities=[fs.Entity("id")],
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(departments_set, departments)
        employees_set = fs.FeatureSet(
            "employees",
            entities=[fs.Entity("id")],
            relations={"departments": {"department_id": "id"}},
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(employees_set, employees)

        features = ["employees.name as n", "departments.name as n2"]

        vector = fs.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp_2 = fs.get_offline_features(
            vector,
            join_type="inner",
            engine_args=engine_args,
        )
        assert_frame_equal(result, resp_2.to_dataframe())

        #  relations according to the argument sent to get offline
        departments_set = fs.FeatureSet("departments", entities=[fs.Entity("id")])
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(departments_set, departments)
        employees_set = fs.FeatureSet("employees", entities=[fs.Entity("id")])
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fs.ingest(employees_set, employees)

        features = ["employees.name as n", "departments.name as n2"]
        relations = {"departments:employees": {"id": "department_id"}}

        vector = fs.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp_3 = fs.get_offline_features(
            vector, join_type="inner", engine_args=engine_args, relations=relations
        )
        assert_frame_equal(result, resp_3.to_dataframe())


def verify_purge(fset, targets):
    fset.reload(update_spec=False)
    orig_status_targets = list(fset.status.targets.keys())
    target_names = [t.name for t in targets]

    for target in fset.status.targets:
        if target.name in target_names:
            driver = get_target_driver(target_spec=target, resource=fset)
            filesystem = driver._get_store().get_filesystem(False)
            if filesystem is not None:
                assert filesystem.exists(driver.get_target_path())

    fset.purge_targets(target_names=target_names)

    for target in fset.status.targets:
        if target.name in target_names:
            driver = get_target_driver(target_spec=target, resource=fset)
            filesystem = driver._get_store().get_filesystem(False)
            assert not filesystem.exists(driver.get_target_path())

    fset.reload(update_spec=False)
    assert set(fset.status.targets.keys()) == set(orig_status_targets) - set(
        target_names
    )


def verify_target_list_fail(targets, with_defaults=None):
    feature_set = fs.FeatureSet(name="target-list-fail", entities=[fs.Entity("ticker")])
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        if with_defaults:
            feature_set.set_targets(targets=targets, with_defaults=with_defaults)
        else:
            feature_set.set_targets(targets=targets)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fs.ingest(feature_set, quotes, targets=targets)


def verify_ingest(
    base_data, keys, infer=False, targets=None, infer_options=fs.InferOptions.default()
):
    if isinstance(keys, str):
        keys = [keys]
    feature_set = fs.FeatureSet("my-feature-set")
    if infer:
        data = base_data.copy()
        fs.preview(feature_set, data, entity_columns=keys)
    else:
        data = base_data.set_index(keys=keys)
    if targets:
        feature_set.set_targets(targets=targets, with_defaults=False)
    df = fs.ingest(feature_set, data, infer_options=infer_options)

    assert len(df) == len(data)
    if infer:
        data.set_index(keys=keys, inplace=True)
    for idx in range(len(df)):
        assert all(df.values[idx] == data.values[idx])


def prepare_feature_set(
    name: str, entity: str, data: pd.DataFrame, timestamp_key=None, targets=None
):
    df_source = mlrun.datastore.sources.DataFrameSource(data, entity, timestamp_key)

    feature_set = fs.FeatureSet(
        name, entities=[fs.Entity(entity)], timestamp_key=timestamp_key
    )
    feature_set.set_targets(targets=targets, with_defaults=False if targets else True)
    df = fs.ingest(feature_set, df_source, infer_options=fs.InferOptions.default())
    return feature_set, df


def map_with_state_test_function(x, state):
    state["sum"] += x["x"]
    return state, state


myfunc = """
def myfunc(context, data):
    print('DATA:', data.artifact_url)
    assert data.meta
    context.log_result('uri', data.artifact_url)
"""
