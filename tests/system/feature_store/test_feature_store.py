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
import mlrun.feature_store as fstore
import tests.conftest
from mlrun.config import config
from mlrun.data_types.data_types import InferOptions, ValueType
from mlrun.datastore.sources import (
    CSVSource,
    DataFrameSource,
    KafkaSource,
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
from mlrun.features import MinMaxValidator, RegexValidator
from mlrun.model import DataTarget
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


class ChangeKey(MapClass):
    def __init__(self, suffix, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix

    def do(self, event):
        if isinstance(event.key, list):
            event.key[-1] += self.suffix
        else:
            event.key += self.suffix

        return event


class IdentityMap(MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, x):
        return x


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
        fset = fstore.FeatureSet(
            "pandass", entities=[fstore.Entity("name")], engine="pandas"
        )
        fstore.ingest(featureset=fset, source=data)

        features = ["pandass.*"]
        vector = fstore.FeatureVector("my-vec", features)
        return vector

    def _ingest_stocks_featureset(self):
        stocks_set = fstore.FeatureSet(
            "stocks", entities=[Entity("ticker", ValueType.STRING)]
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.ingest(
                stocks_set,
                stocks,
                infer_options=fstore.InferOptions.default(),
                run_config=fstore.RunConfig(local=True),
            )

        df = fstore.ingest(
            stocks_set, stocks, infer_options=fstore.InferOptions.default()
        )

        self._logger.info(f"output df:\n{df}")
        stocks_set["name"].description = "some name"

        self._logger.info(f"stocks spec: {stocks_set.to_yaml()}")
        assert (
            stocks_set.spec.features["name"].description == "some name"
        ), "description was not set"
        assert len(df) == len(stocks), "dataframe size doesnt match"
        assert stocks_set.status.stats["exchange"], "stats not created"

    def _ingest_quotes_featureset(self):
        quotes_set = FeatureSet(
            "stock-quotes", entities=["ticker"], timestamp_key="time"
        )

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

        df = fstore.preview(
            quotes_set,
            quotes,
            entity_columns=["ticker"],
            options=fstore.InferOptions.default(),
        )
        self._logger.info(f"quotes spec: {quotes_set.spec.to_yaml()}")
        assert df["zz"].mean() == 9, "map didnt set the zz column properly"
        quotes_set["bid"].validator = MinMaxValidator(min=52, severity="info")

        quotes_set.plot(
            str(self.results_path / "pipe.png"), rankdir="LR", with_targets=True
        )
        df = fstore.ingest(quotes_set, quotes, return_df=True)
        self._logger.info(f"output df:\n{df}")
        assert quotes_set.status.stats.get("asks1_sum_1h"), "stats not created"

    def _get_offline_vector(self, features, features_size, engine=None):
        vector = fstore.FeatureVector("myvector", features, "stock-quotes.xx")
        resp = fstore.get_offline_features(
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
        resp = fstore.get_offline_features(vector.uri, engine=engine)
        df = resp.to_dataframe()
        assert df.shape[1] == features_size, "unexpected num of returned df columns"

    def _get_online_features(self, features, features_size):
        # test real-time query
        vector = fstore.FeatureVector("my-vec", features)
        with fstore.get_online_feature_service(vector) as svc:
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
        stocks_set = fstore.FeatureSet(
            "stocks_parquet_test",
            "stocks set",
            [Entity("ticker", ValueType.STRING)],
            timestamp_key="time",
        )

        df = fstore.ingest(stocks_set, stocks_for_parquet, targets)
        assert len(df) == len(stocks_for_parquet), "dataframe size doesnt match"

        # test get offline features with different parameters
        vector = fstore.FeatureVector("offline-vec", ["stocks_parquet_test.*"])

        # with_indexes = False, entity_timestamp_column = None
        default_df = fstore.get_offline_features(vector).to_dataframe()
        assert isinstance(
            default_df.index, pd.core.indexes.range.RangeIndex
        ), "index column is not of default type"
        assert default_df.index.name is None, "index column is not of default type"
        assert "time" not in default_df.columns, "'time' column shouldn't be present"
        assert (
            "ticker" not in default_df.columns
        ), "'ticker' column shouldn't be present"

        # with_indexes = False, entity_timestamp_column = "time"
        resp = fstore.get_offline_features(vector, entity_timestamp_column="time")
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
        df_with_time = fstore.get_offline_features(
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
        df_with_index = fstore.get_offline_features(vector).to_dataframe()
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
        stocks_set = fstore.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        fstore.ingest(stocks_set, stocks)

        vector = fstore.FeatureVector("SjqevLXR", ["stocks_test.*"])
        target = ParquetTarget(name="parquet", path=target_path)
        if should_raise_error:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fstore.get_offline_features(vector, with_indexes=True, target=target)
        else:
            fstore.get_offline_features(vector, with_indexes=True, target=target)
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
            fstore.ingest(fset, source=stocks, targets=[target])
            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.get_filesystem(False)
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fstore.ingest(fset, source=stocks, targets=[target])

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
            fstore.ingest(fset, source=source, targets=[target])
            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.get_filesystem(False)
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fstore.ingest(fset, source=source, targets=[target])

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
        fset = fstore.FeatureSet("nosql-no-path", entities=[Entity("time_stamp")])
        target_overwrite = True
        ingest_kw = dict()
        if target_overwrite is not None:
            ingest_kw["overwrite"] = target_overwrite
        fstore.ingest(
            fset, df, infer_options=fstore.InferOptions.default(), **ingest_kw
        )

        assert fset.status.targets[
            0
        ].get_path().get_absolute_path() == fset.get_target_path("parquet")
        assert fset.status.targets[
            1
        ].get_path().get_absolute_path() == fset.get_target_path("nosql")

    def test_feature_set_db(self):
        name = "stocks_test"
        stocks_set = fstore.FeatureSet(name, entities=["ticker"])
        fstore.preview(
            stocks_set,
            stocks,
        )
        stocks_set.save()
        db = mlrun.get_run_db()

        sets = db.list_feature_sets(self.project_name, name)
        assert len(sets) == 1, "bad number of results"

        feature_set = fstore.get_feature_set(name, self.project_name)
        assert feature_set.metadata.name == name, "bad feature set response"

        fstore.ingest(stocks_set, stocks)
        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
            fstore.delete_feature_set(name, self.project_name)

        stocks_set.purge_targets()

        fstore.delete_feature_set(name, self.project_name)
        sets = db.list_feature_sets(self.project_name, name)
        assert not sets, "Feature set should be deleted"

    def test_feature_vector_db(self):
        name = "fvec-test"
        fvec = fstore.FeatureVector(name=name)

        db = mlrun.get_run_db()

        # TODO: Using to_dict due to a bug in httpdb api which will be fixed in another PR
        db.create_feature_vector(
            feature_vector=fvec.to_dict(), project=self.project_name
        )

        vecs = db.list_feature_vectors(self.project_name, name)
        assert len(vecs) == 1, "bad number of results"

        feature_vec = fstore.get_feature_vector(name, self.project_name)
        assert feature_vec.metadata.name == name, "bad feature set response"

        fstore.delete_feature_vector(name, self.project_name)
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
        stocks_set = fstore.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        fstore.ingest(stocks_set, stocks)

        vector = fstore.FeatureVector("SjqevLXR", ["stocks_test.*"])
        fstore.get_offline_features(vector)

        actual_stat = vector.get_stats_table().drop("hist", axis=1, errors="ignore")
        actual_stat = actual_stat.sort_index().sort_index(axis=1)
        assert isinstance(actual_stat["top"]["booly"], bool)

    def test_ingest_to_default_path(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )

        fstore.ingest(
            measurements,
            source,
            infer_options=fstore.InferOptions.schema() + fstore.InferOptions.Stats,
            run_config=fstore.RunConfig(local=True),
        )
        final_path = measurements.get_target_path()
        assert "latest" not in final_path
        assert measurements.status.targets is not None
        for target in measurements.status.targets:
            assert "latest" not in target.get_path().get_absolute_path()
            assert target.run_id is not None

    def test_serverless_ingest(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        target_path = os.path.relpath(str(self.results_path / "mycsv.csv"))
        source = CSVSource(
            "mycsv", path=os.path.relpath(str(self.assets_path / "testdata.csv"))
        )
        targets = [CSVTarget("mycsv", path=target_path)]
        if os.path.exists(target_path):
            os.remove(target_path)

        fstore.ingest(
            measurements,
            source,
            targets,
            infer_options=fstore.InferOptions.schema() + fstore.InferOptions.Stats,
            run_config=fstore.RunConfig(local=True),
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

        fset = fstore.FeatureSet(
            name="test", entities=[Entity("patient_id")], timestamp_key="timestamp"
        )
        fstore.ingest(fset, source, targets=[target])

        path_with_runid = path + "/" + fset.status.targets[0].run_id

        list_files = os.listdir(path_with_runid)
        assert len(list_files) == 1 and not os.path.isdir(
            path_with_runid + "/" + list_files[0]
        )
        os.remove(path_with_runid + "/" + list_files[0])

    def test_ingest_with_timestamp(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = CSVSource(
            "mycsv",
            path=os.path.relpath(str(self.assets_path / "testdata.csv")),
            time_field="timestamp",  # TODO: delete this deprecated parameter once it's removed
        )
        resp = fstore.ingest(measurements, source)
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
        source = CSVSource(path=csv_path, parse_dates=["another_time_column"])

        measurements = fstore.FeatureSet(
            "fs", entities=[Entity("key")], timestamp_key="time_stamp"
        )
        try:
            resp = fstore.ingest(measurements, source)
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
        measurements = fstore.FeatureSet(
            "measurements", entities=[Entity(key)], timestamp_key="timestamp"
        )
        source = ParquetSource(
            "myparquet",
            path=os.path.relpath(str(self.assets_path / "testdata.parquet")),
            start_time=datetime(2020, 12, 1, 17, 33, 15),
            end_time="2020-12-01 17:33:16",
        )

        resp = fstore.ingest(
            measurements,
            source,
            return_df=True,
        )
        assert len(resp) == 10

        # start time > timestamp in source
        source = ParquetSource(
            "myparquet",
            path=os.path.relpath(str(self.assets_path / "testdata.parquet")),
            start_time=datetime(2022, 12, 1, 17, 33, 15),
            end_time="2022-12-01 17:33:16",
        )

        resp = fstore.ingest(
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
        measurements = fstore.FeatureSet(
            name, entities=[Entity(key)], timestamp_key="timestamp"
        )
        orig_columns = list(pd.read_csv(str(self.assets_path / "testdata.csv")).columns)
        source = CSVSource(
            "mycsv",
            path=os.path.relpath(str(self.assets_path / "testdata.csv")),
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

        resp1 = fstore.ingest(measurements, source).to_dict()

        features = [
            f"{name}.*",
        ]
        vector = fstore.FeatureVector("myvector", features)
        resp2 = fstore.get_offline_features(
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

        resp = fstore.get_offline_features(
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

    @pytest.mark.parametrize("engine", ["storey", "pandas"])
    def test_passthrough_feature_set(self, engine):
        name = f"measurements_set_{uuid.uuid4()}"
        key = "patient_id"
        measurements_set = fstore.FeatureSet(
            name,
            entities=[Entity(key)],
            timestamp_key="timestamp",
            passthrough=True,
            engine=engine,
        )
        source = CSVSource(
            "mycsv",
            path=os.path.relpath(str(self.assets_path / "testdata.csv")),
            time_field="timestamp",
        )

        expected = source.to_dataframe().set_index("patient_id")

        if engine != "pandas":  # pandas engine does not support preview (ML-2694)
            preview_pd = fstore.preview(
                measurements_set,
                source=source,
            )
            # preview does not do set_index on the entity
            preview_pd.set_index("patient_id", inplace=True)
            assert_frame_equal(expected, preview_pd, check_like=True, check_dtype=False)

        fstore.ingest(measurements_set, source)

        # assert that online target exist (nosql) and offline target does not (parquet)
        assert len(measurements_set.status.targets) == 1
        assert isinstance(measurements_set.status.targets["nosql"], DataTarget)

        # verify that get_offline (and preview) equals the source
        vector = fstore.FeatureVector("myvector", features=[f"{name}.*"])
        resp = fstore.get_offline_features(
            vector, entity_timestamp_column="timestamp", with_indexes=True
        )
        get_offline_pd = resp.to_dataframe()
        get_offline_pd["timestamp"] = pd.to_datetime(get_offline_pd["timestamp"])

        assert_frame_equal(expected, get_offline_pd, check_like=True, check_dtype=False)

        # assert get_online correctness
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get([{"patient_id": "305-90-1613"}])
            assert resp == [
                {
                    "bad": 95,
                    "department": "01e9fe31-76de-45f0-9aed-0f94cc97bca0",
                    "room": 2,
                    "hr": 220.0,
                    "hr_is_error": False,
                    "rr": 25,
                    "rr_is_error": False,
                    "spo2": 99,
                    "spo2_is_error": False,
                    "movements": 4.614601941071927,
                    "movements_is_error": False,
                    "turn_count": 0.3582583538239813,
                    "turn_count_is_error": False,
                    "is_in_bed": 1,
                    "is_in_bed_is_error": False,
                }
            ]

    def test_ingest_twice_with_nulls(self):
        name = f"test_ingest_twice_with_nulls_{uuid.uuid4()}"
        key = "key"

        measurements = fstore.FeatureSet(
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
        resp1 = fstore.ingest(measurements, source)
        assert resp1.to_dict() == {
            "my_string": {"mykey1": "hello"},
            "my_time": {"mykey1": pd.Timestamp("2019-01-26 14:52:37")},
        }

        features = [
            f"{name}.*",
        ]
        vector = fstore.FeatureVector("myvector", features)
        resp2 = fstore.get_offline_features(vector, with_indexes=True)
        resp2 = resp2.to_dataframe()
        assert resp2.to_dict() == {
            "my_string": {"mykey1": "hello"},
            "my_time": {"mykey1": pd.Timestamp("2019-01-26 14:52:37")},
        }

        measurements = fstore.FeatureSet(
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
        resp1 = fstore.ingest(measurements, source, overwrite=False)
        assert resp1.to_dict() == {
            "my_string": {"mykey2": None},
            "my_time": {"mykey2": pd.Timestamp("2019-01-26 14:52:37")},
        }

        features = [
            f"{name}.*",
        ]
        vector = fstore.FeatureVector("myvector", features)
        vector.spec.with_indexes = True
        resp2 = fstore.get_offline_features(vector)
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
        feature_vector = fstore.FeatureVector(
            "test_fv", features, description="test FV"
        )
        res = fstore.get_offline_features(
            feature_vector, entity_timestamp_column="time"
        )
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
        feature_vector = fstore.FeatureVector(
            "test_fv", features, description="test FV"
        )
        res = fstore.get_offline_features(
            feature_vector, entity_timestamp_column="time"
        )
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
        feature_vector = fstore.FeatureVector(
            "test_fv", features, description="test FV"
        )
        res = fstore.get_offline_features(
            feature_vector, entity_timestamp_column="time"
        )
        res = res.to_dataframe()
        assert res.shape[0] == left.shape[0]

    def test_read_csv(self):
        from storey import CSVSource, ReduceToDataFrame, build_flow

        csv_path = str(self.results_path / _generate_random_name() / ".csv")
        targets = [CSVTarget("mycsv", path=csv_path)]
        stocks_set = fstore.FeatureSet(
            "tests", entities=[Entity("ticker", ValueType.STRING)]
        )
        fstore.ingest(
            stocks_set,
            stocks,
            infer_options=fstore.InferOptions.default(),
            targets=targets,
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
        data_set = fstore.FeatureSet(
            name,
            entities=[Entity("first_name"), Entity("last_name")],
            timestamp_key="time",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )
        fstore.preview(
            data_set,
            source=data,
            entity_columns=["first_name", "last_name"],
            options=fstore.InferOptions.default(),
        )

        data_set.plot(
            str(self.results_path / "pipe.png"), rankdir="LR", with_targets=True
        )
        fstore.ingest(data_set, data, return_df=True)

        features = [
            f"{name}.bid_sum_1h",
        ]

        vector = fstore.FeatureVector("my-vec", features)
        with fstore.get_online_feature_service(vector) as svc:
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
        data_set = fstore.FeatureSet("fs4", entities=[Entity("first_name")])

        df = fstore.ingest(data_set, data, return_df=True)

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
        data_set1 = fstore.FeatureSet("fs1", entities=[Entity("string")])
        targets = [ParquetTarget(partitioned=False), NoSqlTarget()]
        fstore.ingest(
            data_set1,
            data,
            targets=targets,
            infer_options=fstore.InferOptions.default(),
        )
        features = ["fs1.*"]
        vector = fstore.FeatureVector("vector", features)
        vector.spec.with_indexes = True

        resp = fstore.get_offline_features(
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

        data_set1 = fstore.FeatureSet("fs1", entities=[Entity("string")])
        targets = [ParquetTarget(partitioned=False), NoSqlTarget()]
        fstore.ingest(
            data_set1,
            data,
            targets=targets,
            infer_options=fstore.InferOptions.default(),
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

        data_set2 = fstore.FeatureSet("fs2", entities=[Entity("string")])
        fstore.ingest(data_set2, data2, infer_options=fstore.InferOptions.default())

        features = ["fs2.data", "fs1.time_stamp"]

        vector = fstore.FeatureVector("vector", features)
        resp = fstore.get_offline_features(
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
        data_set = fstore.FeatureSet(name, entities=[Entity("first_name")])

        data_set.add_aggregation(
            name="bids",
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        fstore.ingest(data_set, data, return_df=True)

        features = [f"{name}.bids_sum_1h", f"{name}.last_name"]

        vector = fstore.FeatureVector("my-vec", features)
        with fstore.get_online_feature_service(vector) as svc:
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
        fset_name = f"{engine}fs"
        fset = fstore.FeatureSet(
            fset_name, entities=[fstore.Entity("name")], engine=engine
        )
        fstore.ingest(featureset=fset, source=data)

        features = [f"{fset_name}.*"]
        vector = fstore.FeatureVector("my-vec", features)
        svc = fstore.get_online_feature_service(vector)
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
        data_set = fstore.FeatureSet("sched_data", entities=[Entity("first_name")])
        fstore.ingest(data_set, data, targets=[data_target])

        path = data_set.status.targets[0].path.format(
            run_id=data_set.status.targets[0].run_id
        )
        assert path == data_set.get_target_path()

        source = ParquetSource(
            "myparquet",
            path=path,
            schedule="mock",
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
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

        feature_set.set_targets(targets, with_defaults=False)

        fstore.ingest(
            feature_set,
            source,
        )

        features = [f"{name}.*"]
        vec = fstore.FeatureVector("sched_test-vec", features)

        svc = fstore.get_online_feature_service(vec)
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
            fstore.ingest(data_set, data, targets=[data_target], overwrite=False)

            fstore.ingest(
                feature_set,
                source,
            )

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
        resp = fstore.get_offline_features(vec)
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
        data_set = fstore.FeatureSet("data", entities=[Entity("first_name")])
        fstore.ingest(data_set, data, targets=[target2])

        path = data_set.status.targets[0].get_path().get_absolute_path()

        source = ParquetSource("myparquet", path=path, schedule="mock")

        feature_set = fstore.FeatureSet(
            name="overwrite",
            entities=[fstore.Entity("first_name")],
            timestamp_key="time",
        )

        targets = [ParquetTarget(path="v3io:///bigdata/bla.parquet", partitioned=False)]

        fstore.ingest(
            feature_set,
            source,
            overwrite=True,
            run_config=fstore.RunConfig(local=False).apply(mlrun.mount_v3io()),
            targets=targets,
        )

        features = ["overwrite.*"]
        vec = fstore.FeatureVector("svec", features)

        # check offline
        resp = fstore.get_offline_features(vec)
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
        data_set = fstore.FeatureSet(
            name, timestamp_key="time", entities=[Entity("first_name")]
        )

        data_set.add_aggregation(
            name="bids",
            column="bid",
            operations=["sum", "max"],
            windows="24h",
        )

        fstore.ingest(data_set, data, return_df=True)

        features = [f"{name}.bids_sum_24h", f"{name}.last_name"]

        vector = fstore.FeatureVector("my-vec", features)
        with fstore.get_online_feature_service(
            vector, fixed_window_type=fixed_window_type
        ) as svc:
            resp = svc.get([{"first_name": "moshe"}])
            if fixed_window_type == FixedWindowType.CurrentOpenWindow:
                expected = {"bids_sum_24h": 2000.0, "last_name": "cohen"}
            else:
                expected = {"bids_sum_24h": 100.0, "last_name": "cohen"}
            assert resp[0] == expected

    def test_split_graph(self):
        quotes_set = fstore.FeatureSet(
            "stock-quotes", entities=[fstore.Entity("ticker")]
        )

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
            fstore.preview(quotes_set, quotes)

        non_default_target_name = "side-target"
        quotes_set.set_targets(
            targets=[
                CSVTarget(name=non_default_target_name, after_state=side_step_name)
            ],
            default_final_step="FeaturesetValidator",
        )

        quotes_set.plot(with_targets=True)

        inf_out = fstore.preview(quotes_set, quotes)
        ing_out = fstore.ingest(quotes_set, quotes, return_df=True)

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
        data_set = fstore.FeatureSet("tests2", entities=[Entity("first_name")])
        fstore.ingest(data_set, data, return_df=True)
        features = ["tests2.*"]
        vector = fstore.FeatureVector("my-vec", features)
        with fstore.get_online_feature_service(vector) as svc:
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
        data_set = fstore.FeatureSet("aliass", entities=[Entity("ticker")])

        data_set.add_aggregation(
            column="price",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        fstore.ingest(data_set, df)
        features = [
            "aliass.price_sum_1h",
            "aliass.price_max_1h as price_m",
        ]
        vector_name = "stocks-vec"
        vector = fstore.FeatureVector(vector_name, features)

        resp = fstore.get_offline_features(vector).to_dataframe()
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

        svc = fstore.get_online_feature_service(vector)
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

        svc = fstore.get_online_feature_service(vector_name)
        try:
            resp = svc.get(entity_rows=[{"ticker": "GOOG"}])
            assert resp[0] == {"price_s": 1441.69, "price_m": 720.92}
        finally:
            svc.close()

        vector = db.get_feature_vector(vector_name, self.project_name, tag="latest")
        stats = vector.get_stats_table()
        assert len(stats) == 2
        assert "price_s" in stats.index

        resp = fstore.get_offline_features(vector).to_dataframe()
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
        csv_vec = fstore.FeatureVector("csv-align-vector", features)
        resp = fstore.get_offline_features(csv_vec)
        csv_vec_df = resp.to_dataframe()

        targets = [ParquetTarget(partitioned=False)]
        parquet_align_set, _ = prepare_feature_set(
            "parquet-align", "ticker", quotes, timestamp_key="time", targets=targets
        )
        parquet_df = parquet_align_set.to_dataframe()
        features = ["parquet-align.*"]
        parquet_vec = fstore.FeatureVector("parquet-align-vector", features)
        resp = fstore.get_offline_features(parquet_vec)
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
            measurements = fstore.FeatureSet(
                name, entities=[Entity(key)], timestamp_key="timestamp"
            )
            if partitioned:
                measurements_partitioned = measurements
            else:
                measurements_nonpartitioned = measurements

            source = CSVSource(
                "mycsv",
                path=os.path.relpath(str(self.assets_path / "testdata.csv")),
            )

            fstore.ingest(
                measurements, source, targets=[ParquetTarget(partitioned=partitioned)]
            )

        columns = ["department", "room"] if with_columns else None
        df_from_partitioned = measurements_partitioned.to_dataframe(columns)
        df_from_nonpartitioned = measurements_nonpartitioned.to_dataframe(columns)
        assert df_from_partitioned.equals(df_from_nonpartitioned)

    def test_sync_pipeline(self):
        stocks_set = fstore.FeatureSet(
            "stocks-sync",
            entities=[Entity("ticker", ValueType.STRING)],
            engine="pandas",
        )

        stocks_set.graph.to(name="s1", handler="myfunc1")
        df = fstore.ingest(stocks_set, stocks)
        self._logger.info(f"output df:\n{df}")

        features = list(stocks_set.spec.features.keys())
        assert len(features) == 1, "wrong num of features"
        assert "exchange" not in features, "field was not dropped"
        assert len(df) == len(stocks), "dataframe size doesnt match"

    @pytest.mark.parametrize("with_graph", [True, False])
    def test_sync_pipeline_chunks(self, with_graph):
        myset = fstore.FeatureSet(
            "early_sense",
            entities=[Entity("patient_id")],
            timestamp_key="timestamp",
            engine="pandas",
        )

        csv_file = os.path.relpath(str(self.assets_path / "testdata.csv"))
        chunksize = 20
        source = CSVSource("mycsv", path=csv_file, attributes={"chunksize": chunksize})

        if with_graph:
            myset.graph.to(name="s1", handler="my_func")

        df = fstore.ingest(myset, source)

        features = list(myset.spec.features.keys())
        print(len(features), features)
        print(myset.to_yaml())
        self._logger.info(f"output df:\n{df}")

        reference_df = pd.read_csv(csv_file)
        reference_df = reference_df[0:chunksize].set_index("patient_id")

        # patient_id (index) and timestamp (timestamp_key) are not in features list
        assert features + ["timestamp"] == list(reference_df.columns)
        assert df.equals(reference_df), "output dataframe is different from expected"

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
        quotes_set = fstore.FeatureSet(
            "post-aggregation", entities=[fstore.Entity("ticker")]
        )
        agg_step = quotes_set.add_aggregation("ask", ["sum", "max"], "1h", "10m")
        agg_step.to("MyMap", "somemap1", field="multi1", multiplier=3)

        # Make sure the map step was added right after the aggregation step
        assert len(quotes_set.graph.steps) == 2
        assert quotes_set.graph.steps[aggregates_step].after == []
        assert quotes_set.graph.steps["somemap1"].after == [aggregates_step]

    def test_featureset_uri(self):
        stocks_set = fstore.FeatureSet("stocks01", entities=[fstore.Entity("ticker")])
        stocks_set.save()
        fstore.ingest(stocks_set.uri, stocks)

    def test_overwrite(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})

        fset = fstore.FeatureSet(name="overwrite-fs", entities=[fstore.Entity("name")])
        targets = [CSVTarget(), ParquetTarget(partitioned=False), NoSqlTarget()]
        fstore.ingest(fset, df1, targets=targets)

        features = ["overwrite-fs.*"]
        fvec = fstore.FeatureVector("overwrite-vec", features=features)

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

        with fstore.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "GHI"}])
            assert resp[0]["value"] == 3

        fstore.ingest(fset, df2, [ParquetTarget(partitioned=False), NoSqlTarget()])

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

        with fstore.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "GHI"}])
            assert resp[0] is None

            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6

    def test_parquet_target_vector_overwrite(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        fset = fstore.FeatureSet(
            name="fvec-parquet-fset", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df1)

        features = ["fvec-parquet-fset.*"]
        fvec = fstore.FeatureVector("fvec-parquet", features=features)
        fvec.spec.with_indexes = True

        target = ParquetTarget()
        off1 = fstore.get_offline_features(fvec, target=target)
        dfout1 = pd.read_parquet(target.get_target_path())

        assert (
            df1.set_index(keys="name")
            .sort_index()
            .equals(off1.to_dataframe().sort_index())
        )
        assert df1.set_index(keys="name").sort_index().equals(dfout1.sort_index())

        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})
        fstore.ingest(fset, df2)
        off2 = fstore.get_offline_features(fvec, target=target)
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

        fset = fstore.FeatureSet(
            name="overwrite-spec-path", entities=[fstore.Entity("name")]
        )
        features = ["overwrite-spec-path.*"]
        fvec = fstore.FeatureVector("overwrite-spec-path-fvec", features=features)

        fstore.ingest(fset, df1, targets=targets)

        fstore.ingest(fset, df2, targets=targets)

        with fstore.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6
            resp = svc.get(entity_rows=[{"name": "ABC"}])
            assert resp[0] is None

    def test_overwrite_single_parquet_file(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})

        targets = [ParquetTarget(path="v3io:///bigdata/overwrite-pq-spec/my.parquet")]

        fset = fstore.FeatureSet(
            name="overwrite-pq-spec-path", entities=[fstore.Entity("name")]
        )

        fstore.ingest(fset, df1, targets=targets)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.ingest(fset, df2, targets=targets, overwrite=False)

    def test_overwrite_false(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})
        df3 = pd.concat([df1, df2])

        fset = fstore.FeatureSet(
            name="override-false", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df1)

        features = ["override-false.*"]
        fvec = fstore.FeatureVector("override-false-vec", features=features)
        fvec.spec.with_indexes = True

        off1 = fstore.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        fstore.ingest(fset, df2, overwrite=False)

        off2 = fstore.get_offline_features(fvec).to_dataframe()
        assert df3.set_index(keys="name").sort_index().equals(off2.sort_index())

        fstore.ingest(fset, df1, targets=[ParquetTarget()])

        off1 = fstore.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        with fstore.get_online_feature_service(fvec) as svc:
            resp = svc.get(entity_rows=[{"name": "PQR"}])
            assert resp[0]["value"] == 6

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.ingest(fset, df1, targets=[CSVTarget()], overwrite=False)

        fset.set_targets(targets=[CSVTarget()])
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.ingest(fset, df1, overwrite=False)

    def test_purge_v3io(self):
        key = "patient_id"
        fset = fstore.FeatureSet(
            "purge", entities=[Entity(key)], timestamp_key="timestamp"
        )
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
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
        fstore.ingest(fset, source)

        verify_purge(fset, targets)

        fstore.ingest(fset, source)

        targets_to_purge = targets[:-1]

        verify_purge(fset, targets_to_purge)

    @pytest.mark.skipif(
        not mlrun.mlconf.redis.url,
        reason="mlrun.mlconf.redis.url is not set, skipping until testing against real redis",
    )
    @pytest.mark.parametrize("target_redis, ", ["", "redis://:aaa@localhost:6379"])
    def test_purge_redis(self, target_redis):
        key = "patient_id"
        fset = fstore.FeatureSet(
            "purge", entities=[Entity(key)], timestamp_key="timestamp"
        )
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
        )
        targets = [
            CSVTarget(),
            CSVTarget(name="specified-path", path="v3io:///bigdata/csv-purge-test.csv"),
            ParquetTarget(partitioned=True, partition_cols=["timestamp"]),
            RedisNoSqlTarget()
            if target_redis == ""
            else RedisNoSqlTarget(path=target_redis),
        ]
        fset.set_targets(
            targets=targets,
            with_defaults=False,
        )
        fstore.ingest(fset, source)

        verify_purge(fset, targets)

        fstore.ingest(fset, source)

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
        fset = fstore.FeatureSet(
            name="nosqlpurge", entities=[Entity(key)], timestamp_key="timestamp"
        )
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource(
            "mycsv",
            path=path,
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
            fstore.ingest(fset, source)
            self._logger.info(f"purging target {tar.name}")
            verify_purge(fset, test_target)

    def test_ingest_dataframe_index(self):
        orig_df = pd.DataFrame([{"x", "y"}])
        orig_df.index.name = "idx"

        fset = fstore.FeatureSet("myfset", entities=[Entity("idx")])
        fstore.ingest(
            fset,
            orig_df,
            [ParquetTarget()],
            infer_options=fstore.InferOptions.default(),
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

        fset = fstore.FeatureSet(
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
        fstore.ingest(fset, source, targets=[ikjqkfcz])

        features = ["rWQTKqbhje.*"]
        vector = fstore.FeatureVector("WPAyrYux", features)
        vector.spec.with_indexes = True
        resp = fstore.get_offline_features(vector)
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
        myset = FeatureSet(
            f"{fset_name}", entities=[Entity("ticker")], timestamp_key="time"
        )
        fstore.ingest(myset, quotes)
        source = StreamSource(key_field="ticker")
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
        run_config = fstore.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fstore.deploy_ingestion_service(
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

        vector = fstore.FeatureVector("my-vec", [f"{fset_name}.*"])
        with fstore.get_online_feature_service(vector) as svc:
            sleep(5)
            resp = svc.get([{"ticker": "AAPL"}])

        assert resp[0]["bid"] == 300

    def test_get_offline_from_feature_set_with_no_schema(self):
        myset = FeatureSet("fset2", entities=[Entity("ticker")])
        fstore.ingest(myset, quotes, infer_options=InferOptions.Null)
        features = ["fset2.*"]
        vector = fstore.FeatureVector("QVMytLdP", features, with_indexes=True)

        try:
            fstore.get_offline_features(vector)
            assert False
        except mlrun.errors.MLRunInvalidArgumentError:
            pass

    def test_join_with_table(self):
        table_url = "v3io:///bigdata/system-test-project/nosql/test_join_with_table"

        df = pd.DataFrame({"name": ["ABC", "DEF"], "aug": ["1", "2"]})
        fset = fstore.FeatureSet(
            name="test_join_with_table_fset", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df, targets=[NoSqlTarget(path=table_url)])
        run_id = fset.status.targets[0].run_id
        table_url = f"{table_url}/{run_id}"
        df = pd.DataFrame(
            {
                "key": ["mykey1", "mykey2", "mykey3"],
                "foreignkey1": ["AB", "DE", "GH"],
                "foreignkey2": ["C", "F", "I"],
            }
        )

        fset = fstore.FeatureSet("myfset", entities=[Entity("key")])
        fset.set_targets([], with_defaults=False)
        fset.graph.to(
            "storey.JoinWithTable",
            table=table_url,
            _key_extractor="(event['foreignkey1'] + event['foreignkey2'])",
            attributes=["aug"],
            inner_join=True,
        )
        df = fstore.ingest(
            fset, df, targets=[], infer_options=fstore.InferOptions.default()
        )
        assert df.to_dict() == {
            "foreignkey1": {"mykey1": "AB", "mykey2": "DE"},
            "foreignkey2": {"mykey1": "C", "mykey2": "F"},
            "aug": {"mykey1": "1", "mykey2": "2"},
        }

    # ML-1167
    def test_directional_graph(self):
        table_url = "v3io:///bigdata/system-test-project/nosql/test_directional_graph"

        df = pd.DataFrame({"name": ["ABC", "DEF"], "aug": ["1", "2"]})
        fset = fstore.FeatureSet(
            name="test_directional_graph", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df, targets=[NoSqlTarget(path=table_url)])
        run_id = fset.status.targets[0].run_id
        table_url = f"{table_url}/{run_id}"
        df = pd.DataFrame(
            {
                "key": ["mykey1", "mykey2", "mykey3"],
                "foreignkey1": ["AB", "DE", "GH"],
                "foreignkey2": ["C", "F", "I"],
            }
        )

        fset = fstore.FeatureSet("myfset", entities=[Entity("key")])
        fset.set_targets([], with_defaults=False)

        fset.graph.to(ChangeKey("_1"), "change1", full_event=True)
        fset.graph.to(ChangeKey("_2"), "change2", full_event=True)
        fset.graph.final_step = "join"

        fset.graph.add_step(
            "storey.JoinWithTable",
            name="join",
            after=["change1", "change2"],
            table=table_url,
            _key_extractor="(event['foreignkey1'] + event['foreignkey2'])",
            attributes=["aug"],
            inner_join=True,
        )
        df = fstore.ingest(fset, df, targets=[])
        assert df.to_dict() == {
            "foreignkey1": {
                "mykey1_1": "AB",
                "mykey1_2": "AB",
                "mykey2_1": "DE",
                "mykey2_2": "DE",
            },
            "foreignkey2": {
                "mykey1_1": "C",
                "mykey1_2": "C",
                "mykey2_1": "F",
                "mykey2_2": "F",
            },
            "aug": {"mykey1_1": "1", "mykey1_2": "1", "mykey2_1": "2", "mykey2_2": "2"},
        }

    def test_get_offline_features_with_tag(self):
        def validate_result(test_vector, test_keys):
            res_set = fstore.get_offline_features(test_vector)
            assert res_set is not None
            res_keys = list(res_set.vector.status.stats.keys())
            assert res_keys.sort() == test_keys.sort()

        data = quotes
        name = "quotes"
        tag = "test"
        project = self.project_name

        test_set = fstore.FeatureSet(
            name, entities=[Entity("ticker", ValueType.STRING)]
        )

        df = fstore.ingest(test_set, data)
        assert df is not None

        # change feature set and save with tag
        test_set.add_aggregation(
            "bid",
            ["avg"],
            "1h",
        )
        new_column = "bid_avg_1h"
        test_set.metadata.tag = tag
        fstore.ingest(test_set, data)

        # retrieve feature set with feature vector and check for changes
        vector = fstore.FeatureVector("vector", [f"{name}.*"], with_indexes=True)
        vector_with_tag = fstore.FeatureVector(
            "vector_with_tag", [f"{name}:{tag}.*"], with_indexes=True
        )
        vector_with_project = fstore.FeatureVector(
            "vector_with_project", [f"{project}/{name}.*"], with_indexes=True
        )
        # vector_with_project.metadata.project = "bs"
        vector_with_features = fstore.FeatureVector(
            "vector_with_features", [f"{name}.bid", f"{name}.time"], with_indexes=True
        )
        vector_with_project_tag_and_features = fstore.FeatureVector(
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
            with fstore.get_online_feature_service(test_vector) as svc:
                sleep(5)
                resp = svc.get([{"ticker": "AAPL"}])
            assert resp is not None
            resp_keys = list(resp[0].keys())
            assert resp_keys.sort() == test_keys.sort()

        data = quotes
        name = "quotes"
        tag = "test"
        project = self.project_name

        test_set = fstore.FeatureSet(
            name, entities=[Entity("ticker", ValueType.STRING)]
        )

        df = fstore.ingest(test_set, data)
        assert df is not None

        # change feature set and save with tag
        test_set.add_aggregation(
            "bid",
            ["avg"],
            "1h",
        )
        new_column = "bid_avg_1h"
        test_set.metadata.tag = tag
        fstore.ingest(test_set, data)

        # retrieve feature set with feature vector and check for changes
        vector = fstore.FeatureVector("vector", [f"{name}.*"], with_indexes=True)
        vector_with_tag = fstore.FeatureVector(
            "vector_with_tag", [f"{name}:{tag}.*"], with_indexes=True
        )
        vector_with_project = fstore.FeatureVector(
            "vector_with_project", [f"{project}/{name}.*"], with_indexes=True
        )
        # vector_with_project.metadata.project = "bs"
        vector_with_features = fstore.FeatureVector(
            "vector_with_features", [f"{name}.bid", f"{name}.time"], with_indexes=True
        )
        vector_with_project_tag_and_features = fstore.FeatureVector(
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
        v3io_source = StreamSource(key_field="ticker")
        fset = fstore.FeatureSet(
            name, entities=[Entity("ticker")], timestamp_key="time"
        )
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

        fstore.preview(
            featureset=fset,
            source=quotes,
            entity_columns=["ticker"],
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
        run_config = fstore.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fstore.deploy_ingestion_service(
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
        vector = fstore.FeatureVector("vecc", features, with_indexes=True)

        fstore.get_offline_features(vector)

    @pytest.mark.parametrize("pass_vector_as_uri", [True, False])
    def test_online_impute(self, pass_vector_as_uri):
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

        data_set1 = fstore.FeatureSet(
            "imp1", entities=[Entity("name")], timestamp_key="time_stamp"
        )
        data_set1.add_aggregation(
            "data",
            ["avg", "max"],
            "1h",
        )
        fstore.ingest(data_set1, data, infer_options=fstore.InferOptions.default())

        data2 = pd.DataFrame({"data2": [1, None, np.inf], "name": ["ab", "cd", "ef"]})

        data_set2 = fstore.FeatureSet("imp2", entities=[Entity("name")])
        fstore.ingest(data_set2, data2, infer_options=fstore.InferOptions.default())

        features = ["imp2.data2", "imp1.data_max_1h", "imp1.data_avg_1h"]

        # create vector and online service with imputing policy
        vector = fstore.FeatureVector("vectori", features)
        vector.save()

        with fstore.get_online_feature_service(
            vector.uri if pass_vector_as_uri else vector,
            impute_policy={"*": "$max", "data_avg_1h": "$mean", "data2": 4},
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
        vector = fstore.FeatureVector("vectori2", features)
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get([{"name": "cd"}])
            assert np.isnan(resp[0]["data2"])
            assert np.isnan(resp[0]["data_avg_1h"])

    def test_map_with_state_with_table(self):
        table_url = (
            "v3io:///bigdata/system-test-project/nosql/test_map_with_state_with_table"
        )

        df = pd.DataFrame({"name": ["a", "b"], "sum": [11, 22]})
        fset = fstore.FeatureSet(
            name="test_map_with_state_with_table_fset", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df, targets=[NoSqlTarget(path=table_url)])
        table_url_with_run_uid = fset.status.targets[0].get_path().get_absolute_path()
        df = pd.DataFrame({"key": ["a", "a", "b"], "x": [2, 3, 4]})

        fset = fstore.FeatureSet("myfset", entities=[Entity("key")])
        fset.set_targets([], with_defaults=False)
        fset.graph.to(
            "storey.MapWithState",
            initial_state=table_url_with_run_uid,
            group_by_key=True,
            _fn="map_with_state_test_function",
        )
        df = fstore.ingest(
            fset, df, targets=[], infer_options=fstore.InferOptions.default()
        )
        assert df.to_dict() == {
            "name": {"a": "a", "b": "b"},
            "sum": {"a": 16, "b": 26},
        }

    def test_get_online_feature_service(self):
        vector = self._generate_vector()
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get([{"name": "ab"}])
            assert resp[0] == {"data": 10}

    def test_allow_empty_vector(self):
        # test that we can pass an non materialized vector to function using special flag
        vector = fstore.FeatureVector("dummy-vec", [])
        vector.save()

        func = mlrun.new_function("myfunc", kind="job", handler="myfunc").with_code(
            body=myfunc
        )
        func.spec.allow_empty_resources = True
        run = func.run(inputs={"data": vector.uri}, local=True)
        assert run.output("uri") == vector.uri

    def test_two_ingests(self):
        df1 = pd.DataFrame({"name": ["AB", "CD"], "some_data": [10, 20]})
        set1 = fstore.FeatureSet("set1", entities=[Entity("name")])
        fstore.ingest(set1, df1)

        df2 = pd.DataFrame({"name": ["AB", "CD"], "some_data": ["Paris", "Tel Aviv"]})
        set2 = fstore.FeatureSet("set2", entities=[Entity("name")])
        fstore.ingest(set2, df2)
        vector = fstore.FeatureVector("check", ["set1.*", "set2.some_data as ddata"])
        svc = fstore.get_online_feature_service(vector)

        try:
            resp = svc.get([{"name": "AB"}])
        finally:
            svc.close()
        assert resp == [{"some_data": 10, "ddata": "Paris"}]

        resp = fstore.get_offline_features(vector)
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
        fset = FeatureSet(
            f"{fset_name}", entities=[Entity("ticker")], timestamp_key="time"
        )

        if feature_set_targets:
            fset.set_targets(feature_set_targets, with_defaults=False)
        fstore.ingest(fset, quotes)
        source = StreamSource(key_field="ticker")
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
        run_config = fstore.RunConfig(function=function, local=False).apply(
            mlrun.mount_v3io()
        )
        fstore.deploy_ingestion_service(
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
        fstore.ingest(feature_set, stocks)
        expected = stocks.to_dict()
        expected.pop("ticker")

        fv = fstore.FeatureVector("fv-label", ["fs-label.*"], "fs-label.name")
        res = fstore.get_offline_features(fv)

        assert res is not None
        assert res.to_dataframe().to_dict() == expected

    def test_get_offline_for_two_feature_set_with_same_column_name(self):
        # This test is testing that all columns are returned with no failure even though
        # two features sets and the label column has the column 'name'.
        expected = ["fs1_exchange", "name", "fs2_name", "fs2_exchange"]

        feature_set = FeatureSet("fs1", entities=[Entity("ticker")])
        fstore.ingest(feature_set, stocks)
        feature_set = FeatureSet("fs2", entities=[Entity("ticker")])
        fstore.ingest(feature_set, stocks)

        fv = fstore.FeatureVector(
            "fv-label", ["fs1.* as fs1", "fs2.* as fs2"], "fs1.name"
        )
        res = fstore.get_offline_features(fv)

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
            dask_cluster.with_worker_requests(mem="2G")
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
        fstore.ingest(data_set, data, infer_options=fstore.InferOptions.default())

        fv_name = "new-fv"
        features = [
            "fs-new.name",
            "fs-new.age",
            "fs-new.department_RD",
            "fs-new.department_IT",
            "fs-new.department_Marketing",
        ]

        my_fv = fstore.FeatureVector(fv_name, features, description="my feature vector")
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
        result_1 = fstore.get_offline_features(
            fv_name,
            target=ParquetTarget(),
            query="age>6 and department_RD==1",
            engine=engine,
            engine_args=engine_args,
        )
        df_res_1 = result_1.to_dataframe()

        assert df_res_1.equals(expected_df)

        result_2 = fstore.get_offline_features(
            fv_name,
            target=ParquetTarget(),
            query="name in ['C']",
            engine=engine,
            engine_args=engine_args,
        )
        df_res_2 = result_2.to_dataframe()

        assert df_res_2.equals(expected_df)

    @pytest.mark.parametrize("engine", ["pandas", "storey"])
    def test_set_event_with_spaces_or_hyphens(self, engine):

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
            "test", entities=[Entity("id")], description="feature set", engine=engine
        )

        data_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        data_set.set_targets()

        df_res = fstore.ingest(
            data_set, data, infer_options=fstore.InferOptions.default()
        )

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

        df_res = fstore.ingest(
            data_set, data, infer_options=fstore.InferOptions.default()
        )

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
            fstore.ingest(data_set, data, infer_options=fstore.InferOptions.default())

    @pytest.mark.skipif(
        not kafka_brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS must be set"
    )
    def test_kafka_target(self, kafka_consumer):

        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fstore.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        target = KafkaTarget(
            "kafka",
            path=kafka_topic,
            bootstrap_servers=kafka_brokers,
        )
        fstore.ingest(stocks_set, stocks, [target])

        expected_records = [
            b'{"ticker": "MSFT", "name": "Microsoft Corporation", "booly": true}',
            b'{"ticker": "GOOG", "name": "Alphabet Inc", "booly": false}',
            b'{"ticker": "AAPL", "name": "Apple Inc", "booly": true}',
        ]

        kafka_consumer.subscribe([kafka_topic])
        for expected_record in expected_records:
            record = next(kafka_consumer)
            assert record.value == expected_record

    @pytest.mark.skipif(
        not kafka_brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS must be set"
    )
    def test_kafka_target_bad_kafka_options(self):

        stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "booly": [True, False, True],
            }
        )
        stocks_set = fstore.FeatureSet(
            "stocks_test", entities=[Entity("ticker", ValueType.STRING)]
        )
        target = KafkaTarget(
            "kafka",
            path=kafka_topic,
            bootstrap_servers=kafka_brokers,
            producer_options={"compression_type": "invalid value"},
        )
        try:
            fstore.ingest(stocks_set, stocks, [target])
            pytest.fail("Expected a ValueError to be raised")
        except ValueError as ex:
            if str(ex) != "Not supported codec: invalid value":
                raise ex

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

        stocks_set = fstore.FeatureSet("stocks", entities=[fstore.Entity("ticker")])
        fstore.ingest(stocks_set, stocks, infer_options=fstore.InferOptions.default())

        quotes_set = fstore.FeatureSet(
            "stock-quotes", entities=[fstore.Entity("ticker")]
        )

        quotes_set.graph.to("storey.Extend", _fn="({'extra': event['bid'] * 77})").to(
            "storey.Filter", "filter", _fn="(event['bid'] > 51.92)"
        ).to(FeaturesetValidator())

        quotes_set.add_aggregation("asks1", ["sum", "max"], "1h", "10m")
        quotes_set.add_aggregation("asks5", ["sum", "max"], "5h", "10m")
        quotes_set.add_aggregation("bids", ["min", "max"], "1h", "10m")

        quotes_set["bid"] = fstore.Feature(
            validator=MinMaxValidator(min=52, severity="info")
        )

        quotes_set.set_targets()

        fstore.preview(
            quotes_set,
            quotes,
            entity_columns=["ticker"],
            timestamp_key="time",
            options=fstore.InferOptions.default(),
        )

        fstore.ingest(quotes_set, quotes)

        features = [
            "stock-quotes.asks5_sum_5h as total_ask",
            "stock-quotes.bids_min_1h",
            "stock-quotes.bids_max_1h",
            "stocks.*",
        ]

        vector_name = "stocks-vec"

        vector = fstore.FeatureVector(
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

        service = fstore.get_online_feature_service(vector_name)
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

    # regression test for #2424
    def test_pandas_write_parquet(self):
        prediction_set = fstore.FeatureSet(
            name="myset", entities=[fstore.Entity("id")], engine="pandas"
        )

        df = pd.DataFrame({"id": ["a", "b"], "number": [11, 22]})

        with tempfile.TemporaryDirectory() as tempdir:
            outdir = f"{tempdir}/test_pandas_write_parquet/"
            prediction_set.set_targets(
                with_defaults=False, targets=[ParquetTarget(path=outdir)]
            )

            returned_df = fstore.ingest(prediction_set, df)

            read_back_df = pd.read_parquet(outdir)
            assert read_back_df.equals(returned_df)

            expected_df = pd.DataFrame({"number": [11, 22]}, index=["a", "b"])
            assert read_back_df.equals(expected_df)

    # regression test for #2557
    @pytest.mark.parametrize(
        ["index_columns"],
        [[["mystr1"]], [["mystr1", "mystr2"]], [["mystr1", "mystr2", "myfloat1"]]],
    )
    def test_pandas_stats_include_index(self, index_columns):
        fset = fstore.FeatureSet(
            "myset",
            entities=[Entity(index_column) for index_column in index_columns],
            engine="pandas",
        )

        fset.graph.to("IdentityMap")

        assert not fset.get_stats_table()

        source_df = pd.DataFrame(
            {
                "mystr1": {0: "ozqqyhvprlghypgn", 1: "etkpkrbuhprigrtk"},
                "mystr2": {0: "kllnbgkcskdiqrqy", 1: "luqsritvfwnfgziw"},
                "myfloat1": {0: 7173728554904657, 1: -8019470409809931},
                "myfloat2": {0: 0.03638798909492902, 1: 0.13661189704381071},
            }
        )
        fstore.preview(fset, source_df)
        actual_stat = fset.get_stats_table().drop("hist", axis=1)
        actual_stat = actual_stat.sort_index().sort_index(axis=1)

        expected_stat = pd.DataFrame(
            {
                "25%": {
                    "myfloat1": -4221170668631284.0,
                    "myfloat2": 0.06144396608214944,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "50%": {
                    "myfloat1": -422870927452637.0,
                    "myfloat2": 0.08649994306936987,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "75%": {
                    "myfloat1": 3375428813726010.0,
                    "myfloat2": 0.11155592005659029,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "count": {
                    "myfloat1": 2.0,
                    "myfloat2": 2.0,
                    "mystr1": 2.0,
                    "mystr2": 2.0,
                },
                "freq": {
                    "myfloat1": math.nan,
                    "myfloat2": math.nan,
                    "mystr1": 1.0,
                    "mystr2": 1.0,
                },
                "max": {
                    "myfloat1": 7173728554904657.0,
                    "myfloat2": 0.13661189704381071,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "mean": {
                    "myfloat1": -422870927452637.0,
                    "myfloat2": 0.08649994306936987,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "min": {
                    "myfloat1": -8019470409809931.0,
                    "myfloat2": 0.03638798909492902,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "std": {
                    "myfloat1": 1.0743214015866118e16,
                    "myfloat2": 0.07086900494767057,
                    "mystr1": math.nan,
                    "mystr2": math.nan,
                },
                "top": {
                    "myfloat1": math.nan,
                    "myfloat2": math.nan,
                    "mystr1": "ozqqyhvprlghypgn",
                    "mystr2": "kllnbgkcskdiqrqy",
                },
                "unique": {
                    "myfloat1": math.nan,
                    "myfloat2": math.nan,
                    "mystr1": 2.0,
                    "mystr2": 2.0,
                },
            },
            index=["myfloat1", "myfloat2", "mystr1", "mystr2"],
        )

        assert_frame_equal(expected_stat, actual_stat)

    @pytest.mark.parametrize("with_indexes", [True, False])
    @pytest.mark.parametrize("engine", ["local", "dask"])
    @pytest.mark.parametrize("join_type", ["inner", "outer"])
    def test_relation_join(self, engine, join_type, with_indexes):
        """Test 3 option of using get offline feature with relations"""
        departments = pd.DataFrame(
            {
                "d_id": [i for i in range(1, 11, 2)],
                "name": [f"dept{num}" for num in range(1, 11, 2)],
                "manager_id": [i for i in range(10, 15)],
            }
        )

        employees_with_department = pd.DataFrame(
            {
                "id": [num for num in range(100, 600, 100)],
                "name": [f"employee{num}" for num in range(100, 600, 100)],
                "department_id": [1, 1, 2, 6, 11],
            }
        )

        employees_with_class = pd.DataFrame(
            {
                "id": [100, 200, 300],
                "name": [f"employee{num}" for num in range(100, 400, 100)],
                "department_id": [1, 1, 2],
                "class_id": [i for i in range(20, 23)],
            }
        )

        classes = pd.DataFrame(
            {
                "c_id": [i for i in range(20, 30, 2)],
                "name": [f"class{num}" for num in range(20, 30, 2)],
            }
        )

        managers = pd.DataFrame(
            {
                "m_id": [i for i in range(10, 20, 2)],
                "name": [f"manager{num}" for num in range(10, 20, 2)],
            }
        )

        join_employee_department = pd.merge(
            employees_with_department,
            departments,
            how=join_type,
            left_on=["department_id"],
            right_on=["d_id"],
            suffixes=("_employees", "_departments"),
        )

        join_employee_managers = pd.merge(
            join_employee_department,
            managers,
            how=join_type,
            left_on=["manager_id"],
            right_on=["m_id"],
            suffixes=("_manage", "_"),
        )

        join_employee_sets = pd.merge(
            employees_with_department,
            employees_with_class,
            how=join_type,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_employees", "_e_mini"),
        )

        _merge_step = pd.merge(
            join_employee_department,
            employees_with_class,
            how=join_type,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_", "_e_mini"),
        )

        join_all = pd.merge(
            _merge_step,
            classes,
            how=join_type,
            left_on=["class_id"],
            right_on=["c_id"],
            suffixes=("_e_mini", "_cls"),
        )

        col_1 = ["name_employees", "name_departments"]
        col_2 = ["name_employees", "name_departments", "name"]
        col_4 = ["name_employees", "name_departments", "name_e_mini", "name_cls"]
        if with_indexes:
            join_employee_sets.set_index(["id"], inplace=True)
            if engine == "local":
                join_employee_department.set_index(["id", "d_id"], inplace=True)
                join_employee_managers.set_index(["id", "d_id", "m_id"], inplace=True)
                join_all.set_index(["id", "d_id", "c_id"], inplace=True)
            else:
                col_1 = ["id", "name_employees", "d_id", "name_departments"]
                col_2 = [
                    "id",
                    "name_employees",
                    "d_id",
                    "name_departments",
                    "m_id",
                    "name",
                ]
                col_4 = [
                    "id",
                    "name_employees",
                    "d_id",
                    "name_departments",
                    "name_e_mini",
                    "c_id",
                    "name_cls",
                ]

        join_employee_department = join_employee_department[col_1].rename(
            columns={"name_departments": "n2", "name_employees": "n"},
        )

        join_employee_managers = join_employee_managers[col_2].rename(
            columns={
                "name_departments": "n2",
                "name_employees": "n",
                "name": "man_name",
            },
        )

        join_employee_sets = join_employee_sets[
            ["name_employees", "name_e_mini"]
        ].rename(
            columns={"name_employees": "n", "name_e_mini": "mini_name"},
        )

        join_all = join_all[col_4].rename(
            columns={
                "name_employees": "n",
                "name_departments": "n2",
                "name_e_mini": "mini_name",
            },
        )

        engine_args = {}
        if engine == "dask":
            dask_cluster = mlrun.new_function(
                "dask_tests", kind="dask", image="mlrun/ml-models"
            )
            dask_cluster.apply(mlrun.mount_v3io())
            dask_cluster.spec.remote = True
            dask_cluster.with_scheduler_requests(mem="2G")
            dask_cluster.save()
            engine_args = {
                "dask_client": dask_cluster,
                "dask_cluster_uri": dask_cluster.uri,
            }
        # relations according to departments_set relations
        managers_set_entity = fstore.Entity("m_id")
        managers_set = fstore.FeatureSet(
            "managers",
            entities=[managers_set_entity],
        )
        managers_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(managers_set, managers)

        classes_set_entity = fstore.Entity("c_id")
        classes_set = fstore.FeatureSet(
            "classes",
            entities=[classes_set_entity],
        )
        managers_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(classes_set, classes)

        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments",
            entities=[departments_set_entity],
            relations={"manager_id": managers_set_entity},
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(departments_set, departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(employees_set, employees_with_department)

        mini_employees_set = fstore.FeatureSet(
            "mini-employees",
            entities=[employees_set_entity],
            relations={
                "department_id": departments_set_entity,
                "class_id": classes_set_entity,
            },
        )
        mini_employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(mini_employees_set, employees_with_class)

        features = ["employees.name"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp = fstore.get_offline_features(
            vector,
            engine_args=engine_args,
            with_indexes=with_indexes,
            engine=engine,
        )
        if with_indexes:
            expected = pd.DataFrame(employees_with_department, columns=["name", "id"])
            expected.set_index("id", inplace=True)
            assert_frame_equal(expected, resp.to_dataframe())
        else:
            assert_frame_equal(
                pd.DataFrame(employees_with_department, columns=["name"]),
                resp.to_dataframe(),
            )
        features = ["employees.name as n", "departments.name as n2"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp_1 = fstore.get_offline_features(
            vector,
            join_type=join_type,
            engine_args=engine_args,
            with_indexes=with_indexes,
            engine=engine,
        )
        assert_frame_equal(join_employee_department, resp_1.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "managers.name as man_name",
        ]

        vector_2 = fstore.FeatureVector(
            "man-vec", features, description="Employees feature vector"
        )
        vector_2.save()

        resp_2 = fstore.get_offline_features(
            vector_2,
            join_type=join_type,
            engine_args=engine_args,
            with_indexes=with_indexes,
            engine=engine,
        )
        assert_frame_equal(join_employee_managers, resp_2.to_dataframe())

        features = ["employees.name as n", "mini-employees.name as mini_name"]

        vector_3 = fstore.FeatureVector(
            "mini-emp-vec", features, description="Employees feature vector"
        )
        vector_3.save()

        resp_3 = fstore.get_offline_features(
            vector_3,
            join_type=join_type,
            engine_args=engine_args,
            with_indexes=with_indexes,
            engine=engine,
        )
        assert_frame_equal(join_employee_sets, resp_3.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "mini-employees.name as mini_name",
            "classes.name as name_cls",
        ]

        vector_4 = fstore.FeatureVector(
            "four-vec", features, description="Employees feature vector"
        )
        vector_4.save()

        resp_4 = fstore.get_offline_features(
            vector_4,
            join_type=join_type,
            engine_args=engine_args,
            with_indexes=with_indexes,
            engine=engine,
        )
        assert_frame_equal(join_all, resp_4.to_dataframe())

    @pytest.mark.parametrize("with_indexes", [True, False])
    def test_relation_asof_join(self, with_indexes):
        """Test 3 option of using get offline feature with relations"""
        departments = pd.DataFrame(
            {
                "d_id": [i for i in range(1, 11, 2)],
                "name": [f"dept{num}" for num in range(1, 11, 2)],
                "manager_id": [i for i in range(10, 15)],
                "time": [pd.Timestamp(f"2023-01-01 0{i}:00:00") for i in range(5)],
            }
        )

        employees_with_department = pd.DataFrame(
            {
                "id": [num for num in range(100, 600, 100)],
                "name": [f"employee{num}" for num in range(100, 600, 100)],
                "department_id": [1, 1, 2, 6, 11],
                "time": [pd.Timestamp(f"2023-01-01 0{i}:00:00") for i in range(5)],
            }
        )

        join_employee_department = pd.merge_asof(
            employees_with_department,
            departments,
            left_on=["time"],
            right_on=["time"],
            left_by=["department_id"],
            right_by=["d_id"],
            suffixes=("_employees", "_departments"),
        )

        col_1 = ["name_employees", "name_departments"]
        if with_indexes:
            join_employee_department.set_index(["id", "d_id"], inplace=True)
            col_1 = ["time"] + col_1

        join_employee_department = join_employee_department[col_1].rename(
            columns={"name_departments": "n2", "name_employees": "n"},
        )

        # relations according to departments_set relations
        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments", entities=[departments_set_entity], timestamp_key="time"
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(departments_set, departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
            timestamp_key="time",
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(employees_set, employees_with_department)

        features = ["employees.name as n", "departments.name as n2"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        resp_1 = fstore.get_offline_features(
            vector,
            with_indexes=with_indexes,
        )
        assert_frame_equal(join_employee_department, resp_1.to_dataframe())

    def test_ingest_with_kafka_source_fails(self):
        source = KafkaSource(
            brokers="broker_host:9092",
            topics="mytopic",
            group="mygroup",
            sasl_user="myuser",
            sasl_pass="mypassword",
        )

        fset = fstore.FeatureSet("myfset", entities=[fstore.Entity("entity")])

        with pytest.raises(mlrun.MLRunInvalidArgumentError):
            fstore.ingest(
                fset,
                source,
            )

    # ML-3099
    def test_preview_timestamp_key(self):
        name = f"measurements_{uuid.uuid4()}"
        base_time = pd.Timestamp(2021, 1, 1)
        data = pd.DataFrame(
            {
                "time": [
                    base_time,
                    base_time - pd.Timedelta(minutes=1),
                    base_time - pd.Timedelta(minutes=2),
                    base_time - pd.Timedelta(minutes=3),
                    base_time - pd.Timedelta(minutes=4),
                    base_time - pd.Timedelta(minutes=5),
                ],
                "first_name": ["moshe", None, "yosi", "yosi", "moshe", "yosi"],
                "last_name": ["cohen", "levi", "levi", "levi", "cohen", "levi"],
                "bid": [2000, 10, 11, 12, 2500, 14],
            }
        )

        # write to kv
        data_set = fstore.FeatureSet(
            name, entities=[Entity("first_name"), Entity("last_name")]
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )
        res_df = fstore.preview(
            data_set,
            source=data,
            entity_columns=["first_name", "last_name"],
            timestamp_key="time",
            options=fstore.InferOptions.default(),
        )
        expected_df = pd.DataFrame(
            [
                ("moshe", "cohen", 2000.0, 2000.0, base_time, 2000),
                ("yosi", "levi", 11.0, 11.0, base_time - pd.Timedelta(minutes=2), 11),
                ("yosi", "levi", 11.0, 11.0, base_time - pd.Timedelta(minutes=3), 12),
                (
                    "moshe",
                    "cohen",
                    2000.0,
                    2000.0,
                    base_time - pd.Timedelta(minutes=4),
                    2500,
                ),
                ("yosi", "levi", 11.0, 11.0, base_time - pd.Timedelta(minutes=5), 14),
            ],
            columns=[
                "first_name",
                "last_name",
                "bid_sum_1h",
                "bid_max_1h",
                "time",
                "bid",
            ],
        )
        expected_df.set_index(["first_name", "last_name"], inplace=True)
        assert res_df.equals(expected_df), f"unexpected result: {res_df}"

    @pytest.mark.parametrize("engine", ["storey", "pandas"])
    def test_ingest_with_validator_from_uri(self, engine):
        df = pd.DataFrame(
            {
                "key": [1, 2, 3, 4, 5, 6, 7],
                "key1": ["1", "2", "3", "4", "5", "6", "7"],
                "key2": ["C", "F", "I", "W", "X", "J", "K"],
            }
        )

        feature_set = fstore.FeatureSet(
            "myfset", entities=[fstore.Entity("key")], engine=engine
        )
        feature_set["key1"] = fstore.Feature(
            validator=RegexValidator(regex=".[A-Za-z]", severity="info"),
            value_type="str",
        )
        try:
            feature_set["key"] = fstore.Feature(
                validator=RegexValidator(regex=".[A-Za-z]", severity="info"),
                value_type="str",
            )
        except mlrun.errors.MLRunInvalidArgumentError:
            pass  # test equal name for entity and feature

        feature_set.graph.to(
            FeaturesetValidator(),
            name="validator",
            columns=["key1"],
            full_event=True,
        )

        feature_set.save()
        output_path = tempfile.TemporaryDirectory()
        df = fstore.ingest(
            feature_set.uri,
            df,
            targets=[ParquetTarget(path=f"{output_path.name}/temp.parquet")],
        )

        assert isinstance(df, pd.DataFrame)


def verify_purge(fset, targets):
    fset.reload(update_spec=False)
    orig_status_targets = list(fset.status.targets.keys())
    from copy import deepcopy

    orig_status_tar = deepcopy(fset.status.targets)
    target_names = [t.name for t in targets]

    for target in fset.status.targets:
        if target.name in target_names:
            driver = get_target_driver(target_spec=target, resource=fset)
            filesystem = driver._get_store().get_filesystem(False)
            if filesystem is not None:
                assert filesystem.exists(driver.get_target_path())
            else:
                files_list = driver._get_store().listdir(driver.get_target_path())
                assert len(files_list) > 0

    fset.purge_targets(target_names=target_names)

    for target in orig_status_tar:
        if target.name in target_names:
            driver = get_target_driver(target_spec=target, resource=fset)
            filesystem = driver._get_store().get_filesystem(False)
            if filesystem is not None:
                assert not filesystem.exists(driver.get_target_path())
            else:
                files_list = driver._get_store().listdir(driver.get_target_path())
                assert len(files_list) == 0

    fset.reload(update_spec=False)
    assert set(fset.status.targets.keys()) == set(orig_status_targets) - set(
        target_names
    )


def verify_target_list_fail(targets, with_defaults=None):
    feature_set = fstore.FeatureSet(
        name="target-list-fail", entities=[fstore.Entity("ticker")]
    )
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        if with_defaults:
            feature_set.set_targets(targets=targets, with_defaults=with_defaults)
        else:
            feature_set.set_targets(targets=targets)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fstore.ingest(feature_set, quotes, targets=targets)


def verify_ingest(
    base_data,
    keys,
    infer=False,
    targets=None,
    infer_options=fstore.InferOptions.default(),
):
    if isinstance(keys, str):
        keys = [keys]
    feature_set = fstore.FeatureSet("my-feature-set")
    if infer:
        data = base_data.copy()
        fstore.preview(feature_set, data, entity_columns=keys)
    else:
        data = base_data.set_index(keys=keys)
    if targets:
        feature_set.set_targets(targets=targets, with_defaults=False)
    df = fstore.ingest(feature_set, data, infer_options=infer_options)

    assert len(df) == len(data)
    if infer:
        data.set_index(keys=keys, inplace=True)
    for idx in range(len(df)):
        assert all(df.values[idx] == data.values[idx])


def prepare_feature_set(
    name: str, entity: str, data: pd.DataFrame, timestamp_key=None, targets=None
):
    df_source = mlrun.datastore.sources.DataFrameSource(data, entity, timestamp_key)

    feature_set = fstore.FeatureSet(
        name, entities=[fstore.Entity(entity)], timestamp_key=timestamp_key
    )
    feature_set.set_targets(targets=targets, with_defaults=False if targets else True)
    df = fstore.ingest(
        feature_set, df_source, infer_options=fstore.InferOptions.default()
    )
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
