import os

from data_sample import quotes, trades, stocks
from storey import MapClass
from storey.flow import _UnaryFunctionFlow

import mlrun.featurestore as fs
from mlrun.config import config as mlconf
from mlrun.featurestore import FeatureSet, Entity, TargetTypes
from mlrun.featurestore.datatypes import ValueType


def init_store():
    mlconf.dbpath = os.environ['TEST_DBPATH']
    data_prefix = os.environ.get('FEATURESTORE_PATH', "v3io:///users/admin/fs")
    client = fs.store_client(data_prefixes={'parquet': "./store", 'nosql': data_prefix})
    client._default_ingest_targets = [TargetTypes.parquet, TargetTypes.nosql]
    return client


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
    return event['bid'] > 51.96


def test_ingestion():
    client = init_store()

    # add feature set without time column (stock ticker metadata)
    stocks_set = fs.FeatureSet("stocks", entities=[Entity('ticker', ValueType.STRING)])
    resp = client.ingest(stocks_set, stocks, infer_schema=True, with_stats=True)
    print(resp)

    quotes_set = FeatureSet("stock-quotes")
    quotes_set.add_flow_step("map", "MyMap", mul=3)
    quotes_set.add_flow_step("addz", "Extend", _fn="({'z': event['bid'] * 77})")
    quotes_set.add_flow_step("filter", "storey.Filter", _fn="(event['bid'] > 51.92)")
    quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")
    quotes_set.add_aggregation("bids", "bid", ["min", "max"], ["1h"], "10m")

    df = quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )
    print(df)

    print(client.ingest(quotes_set, quotes, return_df=True))
    print(quotes_set.to_yaml())


def test_realtime_query():
    client = init_store()

    features = [
        "stock-quotes:bid",
        "stock-quotes:asks_sum_5h",
        "stock-quotes:ask@mycol",
        "stocks:*",
    ]

    resp = client.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    print(resp.to_dataframe())

    svc = client.get_online_feature_service(features)
    resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
    print(resp)
    resp = svc.get([{"ticker": "AAPL"}])
    print(resp)
    svc.close()


def test_feature_set_db():
    name = "stocks_test"
    client = init_store()
    stocks_set = fs.FeatureSet(name, entities=[Entity('ticker', ValueType.STRING)])
    stocks_set.infer_from_df(stocks)
    print(stocks_set.to_yaml())
    client.save_object(stocks_set)

    print(client.list_feature_sets(name))
    return

    fset = client.get_feature_set(name)
    print(fset)

