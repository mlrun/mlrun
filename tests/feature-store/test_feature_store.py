import os

import pandas as pd
import pytest
from data_sample import quotes, stocks, trades
from storey import MapClass

import mlrun
import mlrun.feature_store as fs
from mlrun.config import config as mlconf
from mlrun.data_types.data_types import ValueType
from mlrun.datastore.sources import CSVSource
from mlrun.datastore.targets import CSVTarget, TargetTypes
from mlrun.feature_store import Entity, FeatureSet
from mlrun.feature_store.steps import FeaturesetValidator
from mlrun.features import MinMaxValidator
from mlrun.utils import logger
from tests.conftest import results, tests_root_directory

local_dir = f"{tests_root_directory}/feature-store/"
results_dir = f"{results}/feature-store/"


def init_store():
    mlconf.dbpath = os.environ["TEST_DBPATH"]


def has_db():
    return "TEST_DBPATH" in os.environ


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_basic_featureset():
    init_store()

    # add feature set without time column (stock ticker metadata)
    stocks_set = fs.FeatureSet("stocks", entities=[Entity("ticker", ValueType.STRING)])
    df = fs.ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())

    logger.info(f"output df:\n{df}")
    stocks_set["name"].description = "some name"

    logger.info(f"stocks spec: {stocks_set.to_yaml()}")
    assert (
        stocks_set.spec.features["name"].description == "some name"
    ), "description was not set"
    assert len(df) == len(stocks), "dataframe size doesnt match"
    assert stocks_set.status.stats["exchange"], "stats not created"


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


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_featureset_column_types():
    init_store()

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


class MyMap(MapClass):
    def __init__(self, multiplier=1, **kwargs):
        super().__init__(**kwargs)
        self._multiplier = multiplier

    def do(self, event):
        event["xx"] = event["bid"] * self._multiplier
        event["zz"] = 9
        return event


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_advanced_featureset():
    init_store()

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
    logger.info(f"quotes spec: {quotes_set.spec.to_yaml()}")
    assert df["zz"].mean() == 9, "map didnt set the zz column properly"
    quotes_set["bid"].validator = MinMaxValidator(min=52, severity="info")

    quotes_set.plot(results_dir + "pipe.png", rankdir="LR", with_targets=True)
    df = fs.ingest(quotes_set, quotes, return_df=True)
    logger.info(f"output df:\n{df}")
    assert quotes_set.status.stats.get("asks_sum_1h"), "stats not created"


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_realtime_query():
    init_store()

    features = [
        "stock-quotes.bid",
        "stock-quotes.asks_sum_5h",
        "stock-quotes.ask as mycol",
        "stocks.*",
    ]
    features_size = (
        len(features) + 1 + 1
    )  # (*) returns 2 features, label adds 1 feature
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
    assert len(vector.status.stats) == features_size, "unexpected num of feature stats"
    assert vector.status.label_column == "xx", "unexpected label_column name"

    df = resp.to_dataframe()
    columns = trades.shape[1] + features_size - 2  # - 2 keys
    assert df.shape[1] == columns, "unexpected num of returned df columns"
    resp.to_parquet(results_dir + "query.parquet")

    # test real-time query
    vector = fs.FeatureVector("my-vec", features)
    svc = fs.get_online_feature_service(vector)

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


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_feature_set_db():
    init_store()

    name = "stocks_test"
    stocks_set = fs.FeatureSet(name, entities=[Entity("ticker", ValueType.STRING)])
    fs.infer_metadata(
        stocks_set, stocks,
    )
    stocks_set.save()
    db = mlrun.get_run_db()

    sets = db.list_feature_sets("", name)
    assert len(sets) == 1, "bad number of results"

    feature_set = db.get_feature_set(name)
    assert feature_set.metadata.name == name, "bad feature set response"

    fs.delete_feature_set(name)
    sets = db.list_feature_sets("", name)
    assert not sets, "Feature set should be deleted"


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_feature_vector_db():
    init_store()

    name = "fvec-test"
    fvec = fs.FeatureVector(name=name)

    db = mlrun.get_run_db()

    # TODO: Using to_dict due to a bug in httpdb api which will be fixed in another PR
    db.create_feature_vector(feature_vector=fvec.to_dict())

    vecs = db.list_feature_vectors("", name)
    assert len(vecs) == 1, "bad number of results"

    feature_vec = db.get_feature_vector(name)
    assert feature_vec.metadata.name == name, "bad feature set response"

    fs.delete_feature_vector(name)
    vecs = db.list_feature_vectors("", name)
    assert not vecs, "Feature vector should be deleted"


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_serverless_ingest():
    init_store()
    key = "patient_id"

    measurements = fs.FeatureSet(
        "measurements", entities=[Entity(key)], timestamp_key="timestamp"
    )
    target_path = os.path.relpath(results_dir + "mycsv.csv")
    source = CSVSource("mycsv", path=os.path.relpath(local_dir + "testdata.csv"))
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

    print(measurements.to_yaml())


def prepare_feature_set(name: str, entity: str, data: pd.DataFrame, timestamp_key=None):
    df_source = mlrun.datastore.sources.DataFrameSource(data, entity, timestamp_key)

    feature_set = fs.FeatureSet(
        name, entities=[fs.Entity(entity)], timestamp_key=timestamp_key
    )
    feature_set.set_targets()
    df = fs.ingest(feature_set, df_source, infer_options=fs.InferOptions.default())
    return feature_set, df


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_ordered_pandas_asof_merge():
    init_store()

    left_set, left = prepare_feature_set("left", "ticker", trades, timestamp_key="time")
    right_set, right = prepare_feature_set(
        "right", "ticker", quotes, timestamp_key="time"
    )

    features = ["left.*", "right.*"]
    feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
    res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
    res = res.to_dataframe()
    assert res.shape[0] == left.shape[0]


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_left_not_ordered_pandas_asof_merge():
    init_store()

    left = trades.sort_values(by="price")

    left_set, left = prepare_feature_set("left", "ticker", left, timestamp_key="time")
    right_set, right = prepare_feature_set(
        "right", "ticker", quotes, timestamp_key="time"
    )

    features = ["left.*", "right.*"]
    feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
    res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
    res = res.to_dataframe()
    assert res.shape[0] == left.shape[0]


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_right_not_ordered_pandas_asof_merge():
    init_store()

    right = quotes.sort_values(by="bid")

    left_set, left = prepare_feature_set("left", "ticker", trades, timestamp_key="time")
    right_set, right = prepare_feature_set(
        "right", "ticker", right, timestamp_key="time"
    )

    features = ["left.*", "right.*"]
    feature_vector = fs.FeatureVector("test_fv", features, description="test FV")
    res = fs.get_offline_features(feature_vector, entity_timestamp_column="time")
    res = res.to_dataframe()
    assert res.shape[0] == left.shape[0]
