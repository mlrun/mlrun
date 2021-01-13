import os

import mlrun
import pandas as pd

from mlrun.featurestore.model.base import DataSource, DataTargetSpec
from mlrun.featurestore.sources import CSVSource
from mlrun.featurestore.steps import FeaturesetValidator

from data_sample import quotes, stocks, trades
from storey import MapClass

from mlrun.featurestore.targets import CSVTarget
from mlrun.utils import logger
import mlrun.featurestore as fs
from mlrun.config import config as mlconf
from mlrun.featurestore import FeatureSet, Entity, run_ingestion_task
from mlrun.featurestore.model.datatypes import ValueType
from mlrun.featurestore.model.validators import MinMaxValidator


def init_store():
    mlconf.dbpath = os.environ["TEST_DBPATH"]


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
    def __init__(self, mul=1, **kwargs):
        super().__init__(**kwargs)
        self._mul = mul

    def do(self, event):
        event["xx"] = event["bid"] * self._mul
        event["zz"] = 9
        return event


def test_advanced_featureset():
    init_store()

    quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])

    flow = quotes_set.graph
    flow.to("MyMap", mul=3).to("storey.Extend", _fn="({'z': event['bid'] * 77})").to(
        "storey.Filter", "filter", _fn="(event['bid'] > 51.92)"
    ).to(FeaturesetValidator())

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

    quotes_set.plot("pipe.png", rankdir="LR", with_targets=True)
    df = fs.ingest(quotes_set, quotes, return_df=True)
    logger.info(f"output df:\n{df}")
    assert quotes_set.status.stats.get("asks_sum_1h"), "stats not created"


def test_realtime_query():
    init_store()

    features = [
        "stock-quotes#bid",
        "stock-quotes#asks_sum_5h",
        "stock-quotes#ask as mycol",
        "stocks#*",
    ]

    resp = fs.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    print(resp.vector.to_yaml())
    print(resp.to_dataframe())
    print(resp.to_parquet("./xx.parquet"))

    vector = fs.FeatureVector("my-vec", features)
    vector.spec.graph.to("storey.Extend", _fn="({'xyw': 88})")
    svc = fs.get_online_feature_service(vector)

    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
    print(resp)
    resp = svc.get([{"ticker": "AAPL"}])
    print(resp)
    svc.close()


def test_feature_set_db():
    init_store()

    name = "stocks_test"
    stocks_set = fs.FeatureSet(name, entities=[Entity("ticker", ValueType.STRING)])
    fs.infer_metadata(
        stocks_set, stocks,
    )
    print(stocks_set.to_yaml())
    stocks_set.save()
    db = mlrun.get_run_db()

    print(db.list_feature_sets(name))

    fset = db.get_feature_set(name)
    print(fset)


def test_serverless_ingest():
    init_store()

    measurements = fs.FeatureSet("measurements1", entities=[Entity("patient_id")])
    src_df = pd.read_csv("measurements1.csv")
    df = fs.infer_metadata(
        measurements,
        src_df,
        timestamp_key="timestamp",
        options=fs.InferOptions.default(),
    )
    print(df.head(5))
    source = CSVSource("mycsv", path="measurements1.csv")
    targets = [CSVTarget("mycsv", path="./mycsv.csv")]

    run_ingestion_task(
        measurements,
        source,
        targets,
        name="tst_ingest",
        infer_options=fs.InferOptions.Null,
        parameters={},
        function=None,
        local=True,
    )
