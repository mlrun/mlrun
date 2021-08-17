import os
import pathlib
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from time import sleep

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.util.testing import assert_frame_equal
from storey import MapClass

import mlrun
import mlrun.feature_store as fs
import tests.conftest
from mlrun.data_types.data_types import ValueType
from mlrun.datastore.sources import (
    CSVSource,
    DataFrameSource,
    ParquetSource,
    StreamSource,
)
from mlrun.datastore.targets import (
    CSVTarget,
    NoSqlTarget,
    ParquetTarget,
    TargetTypes,
    get_default_prefix_for_target,
    get_target_driver,
)
from mlrun.feature_store import Entity, FeatureSet
from mlrun.feature_store.feature_set import aggregates_step
from mlrun.feature_store.feature_vector import FixedWindowType
from mlrun.feature_store.steps import FeaturesetValidator
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


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestFeatureStore(TestMLRunSystem):
    def custom_setup(self):
        pass

    def _ingest_stocks_featureset(self):
        stocks_set = fs.FeatureSet(
            "stocks", entities=[Entity("ticker", ValueType.STRING)]
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
        quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])

        flow = quotes_set.graph
        flow.to("MyMap", multiplier=3).to(
            "storey.Extend", _fn="({'z': event['bid'] * 77})"
        ).to("storey.Filter", "filter", _fn="(event['bid'] > 51.92)").to(
            FeaturesetValidator()
        )

        quotes_set.add_aggregation("asks1", "ask", ["sum", "max"], "1h", "10m")
        quotes_set.add_aggregation("asks2", "ask", ["sum", "max"], "5h", "10m")
        quotes_set.add_aggregation("bids", "bid", ["min", "max"], "1h", "10m")

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

    def _get_offline_vector(self, features, features_size):
        vector = fs.FeatureVector("myvector", features, "stock-quotes.xx")
        resp = fs.get_offline_features(
            vector, entity_rows=trades, entity_timestamp_column="time",
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
        resp.to_parquet(str(self.results_path / "query.parquet"))

        # check simple api without join with other df
        # test the use of vector uri
        vector.save()
        resp = fs.get_offline_features(vector.uri)
        df = resp.to_dataframe()
        assert df.shape[1] == features_size, "unexpected num of returned df columns"

    def _get_online_features(self, features, features_size):
        # test real-time query
        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)
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
        svc.close()

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
        self._get_offline_vector(features, features_size)

        self._logger.debug("Get online feature vector")
        self._get_online_features(features, features_size)

    def test_feature_set_db(self):
        name = "stocks_test"
        stocks_set = fs.FeatureSet(name, entities=[Entity("ticker", ValueType.STRING)])
        fs.preview(
            stocks_set, stocks,
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
        assert os.path.exists(target_path), "result file was not generated"
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
        target = ParquetTarget(path=path)

        fset = fs.FeatureSet(
            name="test", entities=[Entity("patient_id")], timestamp_key="timestamp"
        )
        fs.ingest(fset, source, targets=[target])

        list_files = os.listdir(path)
        assert len(list_files) == 1 and not os.path.isdir(path + "/" + list_files[0])
        os.remove(path + "/" + list_files[0])

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

        csv_path = "/tmp/multiple_time_columns.csv"
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

        resp = fs.ingest(measurements, source, return_df=True,)
        assert len(resp) == 10

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
        resp2 = fs.get_offline_features(vector, entity_timestamp_column="timestamp")
        resp2 = resp2.to_dataframe().to_dict()

        resp1.pop("timestamp")
        assert resp1 == resp2

        file_system = fsspec.filesystem("v3io")
        kind = TargetTypes.parquet
        path = f"{get_default_prefix_for_target(kind)}/sets/{name}-latest"
        path = path.format(name=name, kind=kind, project=self.project_name)
        dataset = pq.ParquetDataset(path, filesystem=file_system,)
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
            time_partitioning_granularity = "hour"
        if time_partitioning_granularity:
            for unit in ["year", "month", "day", "hour"]:
                expected_partitions.append(unit)
                if unit == time_partitioning_granularity:
                    break

        assert partitions == expected_partitions

        resp = fs.get_offline_features(
            vector,
            start_time=datetime(2020, 12, 1, 17, 33, 15),
            end_time=datetime(2020, 12, 1, 17, 33, 16),
            entity_timestamp_column="timestamp",
        )
        resp2 = resp.to_dataframe()
        assert len(resp2) == 10
        result_columns = list(resp2.columns)
        orig_columns.remove("timestamp")
        orig_columns.remove("patient_id")
        assert result_columns == orig_columns

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
            targets=[ParquetTarget(partitioned=True)], with_defaults=False,
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
        resp2 = fs.get_offline_features(vector)
        resp2 = resp2.to_dataframe()
        assert resp2.to_dict() == {"my_string": {"mykey1": "hello"}}

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
            targets=[ParquetTarget(partitioned=True)], with_defaults=False,
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
        resp2 = fs.get_offline_features(vector)
        resp2 = resp2.to_dataframe()
        assert resp2.to_dict() == {"my_string": {"mykey1": "hello", "mykey2": None}}

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
        controller = build_flow([CSVSource(csv_path), ReduceToDataFrame()]).run()
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
        os.remove(csv_path)

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
            name="bids",
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
            f"{name}.bids_sum_1h",
        ]

        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)

        resp = svc.get([{"first_name": "yosi", "last_name": "levi"}])
        assert resp[0]["bids_sum_1h"] == 37.0

        svc.close()

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
        fs.ingest(data_set1, data, infer_options=fs.InferOptions.default())
        features = ["fs1.*"]
        vector = fs.FeatureVector("vector", features)
        vector.spec.with_indexes = True

        resp = fs.get_offline_features(
            vector,
            entity_timestamp_column="time_stamp",
            start_time=datetime(2021, 6, 9, 9, 30),
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
        fs.ingest(data_set1, data, infer_options=fs.InferOptions.default())

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
            end_time=datetime(2021, 6, 9, 10, 30),
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
        svc = fs.get_online_feature_service(vector)

        resp = svc.get([{"first_name": "moshe"}])
        expected = {"bids_sum_1h": 2000.0, "last_name": "cohen"}
        assert resp[0] == expected
        svc.close()

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

    def test_ingest_pandas_engine(self):
        data = pd.DataFrame({"name": ["ab", "cd"], "data": [10, 20]})

        data.set_index(["name"], inplace=True)
        fset = fs.FeatureSet("pandass", entities=[fs.Entity("name")], engine="pandas")
        fs.ingest(featureset=fset, source=data)

        features = ["pandass.*"]
        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)

        resp = svc.get([{"name": "ab"}])
        assert resp[0] == {"data": 10}

        svc.close()

    @pytest.mark.parametrize("partitioned", [True, False])
    def test_schedule_on_filtered_by_time(self, partitioned):
        name = f"sched-time-{str(partitioned)}"

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
        target2 = ParquetTarget()
        data_set = fs.FeatureSet("data", entities=[Entity("first_name")])
        fs.ingest(data_set, data, targets=[target2])

        path = data_set.status.targets[0].path

        # the job will be scheduled every minute
        cron_trigger = "*/1 * * * *"

        source = ParquetSource(
            "myparquet", path=path, time_field="time", schedule=cron_trigger
        )

        feature_set = fs.FeatureSet(
            name=name, entities=[fs.Entity("first_name")], timestamp_key="time",
        )

        if partitioned:
            targets = [
                NoSqlTarget(),
                ParquetTarget(
                    name="tar1",
                    path="v3io:///bigdata/fs1/",
                    partitioned=True,
                    partition_cols=["time"],
                ),
            ]
        else:
            targets = [
                ParquetTarget(
                    name="tar2", path="v3io:///bigdata/fs2/", partitioned=False
                ),
                NoSqlTarget(),
            ]

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False).apply(mlrun.mount_v3io()),
            targets=targets,
        )
        sleep(60)

        features = [f"{name}.*"]
        vec = fs.FeatureVector("sched_test-vec", features)

        svc = fs.get_online_feature_service(vec)

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
        fs.ingest(data_set, data, targets=[target2])

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

        svc.close()

        # check offline
        resp = fs.get_offline_features(vec)
        assert len(resp.to_dataframe() == 4)
        assert "uri" not in resp.to_dataframe() and "katya" not in resp.to_dataframe()

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
            name="bids", column="bid", operations=["sum", "max"], windows="24h",
        )

        fs.ingest(data_set, data, return_df=True)

        features = [f"{name}.bids_sum_24h", f"{name}.last_name"]

        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector, fixed_window_type=fixed_window_type)

        resp = svc.get([{"first_name": "moshe"}])
        if fixed_window_type == FixedWindowType.CurrentOpenWindow:
            expected = {"bids_sum_24h": 2000.0, "last_name": "cohen"}
        else:
            expected = {"bids_sum_24h": 100.0, "last_name": "cohen"}
        assert resp[0] == expected
        svc.close()

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
        svc = fs.get_online_feature_service(vector)

        resp = svc.get([{"first_name": "yossi"}])
        assert resp[0] == {"bid": 10, "bool": None}

        svc.close()

    def test_forced_columns_target(self):
        columns = ["time", "ask"]
        targets = [ParquetTarget(columns=columns)]
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

        targets = [ParquetTarget()]
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
        parquet1 = pd.read_parquet(feature_set.get_target_path(name="parquet1"))
        parquet2 = pd.read_parquet(feature_set.get_target_path(name="parquet2"))

        assert all(parquet1 == quotes.set_index("ticker"))
        assert all(parquet1 == parquet2)

        os.remove(parquet_path1)
        os.remove(parquet_path2)

    def test_post_aggregation_step(self):
        quotes_set = fs.FeatureSet("post-aggregation", entities=[fs.Entity("ticker")])
        agg_step = quotes_set.add_aggregation(
            "asks", "ask", ["sum", "max"], "1h", "10m"
        )
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
        fs.ingest(fset, df1, targets=[CSVTarget(), ParquetTarget(), NoSqlTarget()])

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

        svc = fs.get_online_feature_service(fvec)
        resp = svc.get(entity_rows=[{"name": "GHI"}])
        assert resp[0]["value"] == 3
        svc.close()

        fs.ingest(fset, df2)

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

        svc = fs.get_online_feature_service(fvec)
        resp = svc.get(entity_rows=[{"name": "GHI"}])
        assert resp[0] is None

        resp = svc.get(entity_rows=[{"name": "PQR"}])
        assert resp[0]["value"] == 6
        svc.close()

    def test_parquet_target_vector_overwrite(self):
        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        fset = fs.FeatureSet(name="fvec-parquet-fset", entities=[fs.Entity("name")])
        fs.ingest(fset, df1)

        features = ["fvec-parquet-fset.*"]
        fvec = fs.FeatureVector("fvec-parquet", features=features)

        target = ParquetTarget()
        off1 = fs.get_offline_features(fvec, target=target)
        dfout1 = pd.read_parquet(target._target_path)

        assert (
            df1.set_index(keys="name")
            .sort_index()
            .equals(off1.to_dataframe().sort_index())
        )
        assert df1.set_index(keys="name").sort_index().equals(dfout1.sort_index())

        df2 = pd.DataFrame({"name": ["JKL", "MNO", "PQR"], "value": [4, 5, 6]})
        fs.ingest(fset, df2)
        off2 = fs.get_offline_features(fvec, target=target)
        dfout2 = pd.read_parquet(target._target_path)
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

        svc = fs.get_online_feature_service(fvec)
        resp = svc.get(entity_rows=[{"name": "PQR"}])
        assert resp[0]["value"] == 6
        resp = svc.get(entity_rows=[{"name": "ABC"}])
        assert resp[0] is None
        svc.close()

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

        off1 = fs.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        fs.ingest(fset, df2, overwrite=False)

        off2 = fs.get_offline_features(fvec).to_dataframe()
        assert df3.set_index(keys="name").sort_index().equals(off2.sort_index())

        fs.ingest(fset, df1, targets=[ParquetTarget()])

        off1 = fs.get_offline_features(fvec).to_dataframe()
        assert df1.set_index(keys="name").sort_index().equals(off1.sort_index())

        svc = fs.get_online_feature_service(fvec)
        resp = svc.get(entity_rows=[{"name": "PQR"}])
        assert resp[0]["value"] == 6
        svc.close()

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(fset, df1, targets=[CSVTarget()], overwrite=False)

        fset.set_targets(targets=[CSVTarget()])
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(fset, df1, overwrite=False)

    def test_purge(self):
        key = "patient_id"
        fset = fs.FeatureSet("purge", entities=[Entity(key)], timestamp_key="timestamp")
        path = os.path.relpath(str(self.assets_path / "testdata.csv"))
        source = CSVSource("mycsv", path=path, time_field="timestamp",)
        targets = [
            CSVTarget(),
            CSVTarget(name="specified-path", path="v3io:///bigdata/csv-purge-test.csv"),
            ParquetTarget(partitioned=True, partition_cols=["timestamp"]),
            NoSqlTarget(),
        ]
        fset.set_targets(
            targets=targets, with_defaults=False,
        )
        fs.ingest(fset, source)

        verify_purge(fset, targets)

        fs.ingest(fset, source)

        targets_to_purge = targets[:-1]
        verify_purge(fset, targets_to_purge)

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
        vector.spec.with_indexes = False
        resp = fs.get_offline_features(vector)
        off_df = resp.to_dataframe()
        del orig_df["time_stamp"]
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
        myset = FeatureSet("fset2", entities=[Entity("ticker")])
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
        stream_path = f"v3io:///projects/{function.metadata.project}/FeatureStore/fset2/v3ioStream"
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
        vector = fs.FeatureVector("my-vec", ["fset2.*"])
        svc = fs.get_online_feature_service(vector)
        sleep(5)
        resp = svc.get([{"ticker": "AAPL"}])
        svc.close()
        assert resp[0]["bid"] == 300


def verify_purge(fset, targets):
    fset.reload(update_spec=False)
    orig_status_targets = list(fset.status.targets.keys())
    target_names = [t.name for t in targets]

    for target in targets:
        driver = get_target_driver(target_spec=target, resource=fset)
        filesystem = driver._get_store().get_filesystem(False)
        assert filesystem.exists(driver._target_path)

    fset.purge_targets(target_names=target_names)

    for target in targets:
        driver = get_target_driver(target_spec=target, resource=fset)
        filesystem = driver._get_store().get_filesystem(False)
        assert not filesystem.exists(driver._target_path)

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
