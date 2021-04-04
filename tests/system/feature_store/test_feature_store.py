import os
import random
import string
from datetime import datetime

import pandas as pd
import pytest
from storey import MapClass

import mlrun
import mlrun.feature_store as fs
from mlrun.data_types.data_types import ValueType
from mlrun.datastore.sources import CSVSource
from mlrun.datastore.targets import CSVTarget, TargetTypes
from mlrun.feature_store import Entity, FeatureSet
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

        quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")
        quotes_set.add_aggregation("bids", "bid", ["min", "max"], ["1h"], "10m")

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
        assert quotes_set.status.stats.get("asks_sum_1h"), "stats not created"

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

        resp = svc.get([{"ticker": "a"}])
        assert(resp[0] == {None})
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
            "stock-quotes.asks_sum_5h",
            "stock-quotes.ask as mycol",
            "stocks.*",
        ]
        features_size = (
            len(features) + 1 + 1
        )  # (*) returns 2 features, label adds 1 feature
        self._get_offline_vector(features, features_size)

        self._logger.debug("Get offline feature vector")
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

        feature_set = db.get_feature_set(name, self.project_name)
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

        feature_vec = db.get_feature_vector(name, self.project_name)
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
        from storey import ReadCSV, ReduceToDataFrame, build_flow

        csv_path = str(self.results_path / _generate_random_name() / ".csv")
        targets = [CSVTarget("mycsv", path=csv_path)]
        stocks_set = fs.FeatureSet(
            "tests", entities=[Entity("ticker", ValueType.STRING)]
        )
        fs.ingest(
            stocks_set, stocks, infer_options=fs.InferOptions.default(), targets=targets
        )

        # reading csv file
        controller = build_flow([ReadCSV(csv_path), ReduceToDataFrame()]).run()
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
            "tests2", entities=[Entity("first_name"), Entity("last_name")]
        )

        data_set.add_aggregation(
            name="bids",
            column="bid",
            operations=["sum", "max"],
            windows=["1h"],
            period="10m",
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
            "tests2.bids_sum_1h",
        ]

        vector = fs.FeatureVector("my-vec", features)
        svc = fs.get_online_feature_service(vector)

        resp = svc.get([{"first_name": "yosi", "last_name": "levi"}])
        print(resp[0])

        svc.close()


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


def prepare_feature_set(name: str, entity: str, data: pd.DataFrame, timestamp_key=None):
    df_source = mlrun.datastore.sources.DataFrameSource(data, entity, timestamp_key)

    feature_set = fs.FeatureSet(
        name, entities=[fs.Entity(entity)], timestamp_key=timestamp_key
    )
    feature_set.set_targets()
    df = fs.ingest(feature_set, df_source, infer_options=fs.InferOptions.default())
    return feature_set, df
