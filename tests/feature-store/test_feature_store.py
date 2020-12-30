import os

from mlrun.featurestore.steps import FeaturesetValidator

from data_sample import quotes, stocks
from storey import MapClass
from storey.flow import _UnaryFunctionFlow

import mlrun.featurestore as fs
from mlrun.config import config as mlconf
from mlrun.featurestore import FeatureSet, Entity
from mlrun.featurestore.model.datatypes import ValueType
from mlrun.featurestore.model.validators import MinMaxValidator


def init_store():
    mlconf.dbpath = os.environ["TEST_DBPATH"]


class MyMap(MapClass):
    def __init__(self, mul=1, **kwargs):
        super().__init__(**kwargs)
        self._mul = mul

    def do(self, event):
        event["xx"] = event["bid"] * self._mul
        event["zz"] = 9
        return event


class Extend(_UnaryFunctionFlow):
    async def _do_internal(self, event, fn_result):
        for key, value in fn_result.items():
            event.body[key] = value
        await self._do_downstream(event)


def my_filter(event):
    return event["bid"] > 51.96


def test_ingestion():
    init_store()

    # add feature set without time column (stock ticker metadata)
    stocks_set = fs.FeatureSet("stocks", entities=[Entity("ticker", ValueType.STRING)])
    print(stocks_set.spec.graph.to_yaml())
    resp = fs.ingest(stocks_set, stocks, infer_options=fs.InferOptions.default())
    print(resp)

    stocks_set["name"].description = "some name"
    print(stocks_set.to_yaml())

    quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])
    # quotes_set.set_targets()

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
    print(df)
    quotes_set["bid"].validator = MinMaxValidator(min=52, severity="info")

    quotes_set.plot("pipe.png", rankdir="LR", with_targets=True)
    print(quotes_set.to_yaml())

    print(fs.ingest(quotes_set, quotes, return_df=True))
    print(quotes_set.to_yaml())


def test_realtime_query():
    init_store()

    features = [
        "stock-quotes#bid",
        "stock-quotes#asks_sum_5h",
        "stock-quotes#ask as mycol",
        "stocks#*",
    ]

    # resp = fs.get_offline_features(
    #     features, entity_rows=trades, entity_timestamp_column="time"
    # )
    # print(resp.vector.to_yaml())
    # print(resp.to_dataframe())
    # print(resp.to_parquet("./xx.parquet"))

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
    stocks_set.infer_from_df(stocks)
    print(stocks_set.to_yaml())
    stocks_set.save()

    print(fs.list_feature_sets(name))
    return

    fset = fs.get_feature_set(name)
    print(fset)
