import os
import random
import string
import uuid
from datetime import datetime

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import pytest
from storey import EmitAfterMaxEvent, MapClass

import mlrun
import mlrun.feature_store as fs
from mlrun.data_types.data_types import ValueType
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import (
    CSVTarget,
    ParquetTarget,
    TargetTypes,
    get_default_prefix_for_target,
)
from mlrun.feature_store import Entity, FeatureSet
from mlrun.feature_store.feature_set import aggregates_step
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

        df = fs.infer_metadata(
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
        resp = fs.get_offline_features(vector)
        df = resp.to_dataframe()
        assert df.shape[1] == features_size, "unexpected num of returned df columns"

    def _get_online_features(self, features, features_size):
        # test real-time query
        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)
        # check non existing column
        resp = svc.get([{"bb": "AAPL"}])

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
        fs.infer_metadata(
            stocks_set, stocks,
        )
        stocks_set.save()
        db = mlrun.get_run_db()

        sets = db.list_feature_sets(self.project_name, name)
        assert len(sets) == 1, "bad number of results"

        feature_set = fs.get_feature_set(name, self.project_name)
        assert feature_set.metadata.name == name, "bad feature set response"

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
        assert features == stats, "didnt infer stats for all features"

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
        key = "patient_id"
        name = f"measurements_{uuid.uuid4()}"
        measurements = fs.FeatureSet(name, entities=[Entity(key)])
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
        resp1 = fs.ingest(measurements, source)

        features = [
            f"{name}.*",
        ]
        vector = fs.FeatureVector("myvector", features)
        resp = fs.get_offline_features(vector)
        resp2 = resp.to_dataframe()

        assert resp1.to_dict() == resp2.to_dict()

        file_system = fsspec.filesystem("v3io")
        kind = TargetTypes.parquet
        path = f"{get_default_prefix_for_target(kind)}/sets/{name}-latest"
        path = path.format(name=name, kind=kind, project="system-test-project")
        dataset = pq.ParquetDataset(path, filesystem=file_system,)
        partitions = [key for key, _ in dataset.pieces[0].partition_keys]

        if key_bucketing_number is None:
            expected_partitions = []
        elif key_bucketing_number == 0:
            expected_partitions = ["igzpart_key"]
        else:
            expected_partitions = [f"igzpart_hash{key_bucketing_number}_key"]
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
                expected_partitions.append(f"igzpart_{unit}")
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

    def test_ordered_pandas_asof_merge(self):
        left_set, left = prepare_feature_set(
            "left", "ticker", trades, timestamp_key="time"
        )
        right_set, right = prepare_feature_set(
            "right", "ticker", quotes, timestamp_key="time"
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
                "first_name": ["moshe", "yosi", "yosi", "yosi", "moshe", "yosi"],
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
            emit_policy=EmitAfterMaxEvent(1),
        )
        fs.infer_metadata(
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
        assert resp[0]["bids_sum_1h"] == 47.0

        svc.close()

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
            fs.infer_metadata(quotes_set, quotes)

        non_default_target_name = "side-target"
        quotes_set.set_targets(
            targets=[
                CSVTarget(name=non_default_target_name, after_state=side_step_name)
            ],
            default_final_state="FeaturesetValidator",
        )

        quotes_set.plot(with_targets=True)

        inf_out = fs.infer_metadata(quotes_set, quotes)
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
        # df = fs.infer(my_set, df.head())
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
        fs.infer_metadata(feature_set, data, entity_columns=keys)
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
