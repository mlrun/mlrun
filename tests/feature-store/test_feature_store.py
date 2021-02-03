import os

import mlrun
import pytest
import pandas as pd
from tests.conftest import results, tests_root_directory

from mlrun.feature_store.sources import CSVSource
from mlrun.feature_store.steps import FeaturesetValidator

from data_sample import quotes, stocks, trades
from storey import MapClass

from mlrun.feature_store.targets import CSVTarget
from mlrun.utils import logger
import mlrun.feature_store as fs
from mlrun.config import config as mlconf
from mlrun.feature_store import FeatureSet, Entity, run_ingestion_task
from mlrun.data_types import ValueType
from mlrun.features import MinMaxValidator


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
    assert len(df) == len(stocks), "datafreame size doesnt match"
    assert stocks_set.status.stats["exchange"], "stats not created"


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

    resp = fs.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    vector = resp.vector
    assert len(vector.spec.features) == len(
        features
    ), "unexpected num of requested features"
    # stocks (*) returns 2 features
    assert (
        len(vector.status.features) == len(features) + 1
    ), "unexpected num of returned features"
    assert (
        len(vector.status.stats) == len(features) + 1
    ), "unexpected num of feature stats"

    df = resp.to_dataframe()
    columns = trades.shape[1] + len(features) + 1
    assert df.shape[1] == columns, "unexpected num of returned df columns"
    resp.to_parquet(results_dir + "query.parquet")

    # test real-time query
    vector = fs.FeatureVector("my-vec", features)
    svc = fs.get_online_feature_service(vector)

    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
    print(resp)
    resp = svc.get([{"ticker": "AAPL"}])
    assert (
        resp[0]["ticker"] == "AAPL" and resp[0]["exchange"] == "NASDAQ"
    ), "unexpected online result"
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
    assert feature_set.metadata.name == name, "bas feature set response"


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_serverless_ingest():
    init_store()

    measurements = fs.FeatureSet("measurements", entities=[Entity("patient_id")])
    src_df = pd.read_csv(local_dir + "testdata.csv")
    df = fs.infer_metadata(
        measurements,
        src_df,
        timestamp_key="timestamp",
        options=fs.InferOptions.default(),
    )
    print(df.head(5))
    target_path = os.path.relpath(results_dir + "mycsv.csv")
    source = CSVSource("mycsv", path=os.path.relpath(local_dir + "testdata.csv"))
    targets = [CSVTarget("mycsv", path=target_path)]
    if os.path.exists(target_path):
        os.remove(target_path)

    run_ingestion_task(
        measurements,
        source,
        targets,
        name="test_ingest",
        infer_options=fs.InferOptions.Null,
        parameters={},
        function=None,
        local=True,
    )
    assert os.path.exists(target_path), "result file was not generated"
