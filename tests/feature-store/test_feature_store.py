from data_sample import quotes, trades, stocks
from storey import MapClass
import mlrun.featurestore as fs
from mlrun.featurestore import FeatureSet, Entity, TargetTypes
from mlrun.featurestore.datatypes import ValueType


def test_get_offline():
    client = fs.store_client(data_prefix="./store")

    # add feature set with time (for time travel) and record its stats
    quotes_set = fs.FeatureSet("stock-quotes")
    df = quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )

    # save ingest data and print the FeatureSet spec
    client.ingest(quotes_set, quotes)
    print(df)
    print(list(quotes_set.spec.features))
    print(quotes_set.to_yaml())

    # add feature set without time column (stock ticker metadata)
    stocks_set = fs.FeatureSet("stocks")
    stocks_set.infer_from_df(stocks, entity_columns=["ticker"])
    resp = client.ingest(stocks_set, stocks)
    print(resp)

    features = ["stock-quotes:bid", "stock-quotes:bid_sum_5h", "stock-quotes:ask@mycol", "stocks:*"]
    resp = client.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    print(resp.to_dataframe())

    # service = client.get_online_feature_service(features)
    # service.get(trades)


class MyMap(MapClass):
    def __init__(self, mul=1, **kwargs):
        super().__init__(**kwargs)
        self._mul = mul

    def do(self, event):
        event["xx"] = event["bid"] * self._mul
        event["zz"] = 9
        return event


def test_storey():
    client = fs.store_client(data_prefix="v3io:///users/admin/fs")
    client._default_ingest_targets = [TargetTypes.parquet, TargetTypes.nosql]
    # client.nosql_path_prefix = 'users/admin/fs'

    stocks_set = fs.FeatureSet("stocks")
    stocks_set.infer_from_df(stocks, entity_columns=["ticker"])
    resp = client.ingest(stocks_set, stocks)
    print(resp)
    return

    quotes_set = FeatureSet(
        "stock-quotes",
        entities=[Entity("ticker", ValueType.STRING)],
        timestamp_key="time",
    )
    quotes_set.add_flow_step("map", "MyMap", mul=3)
    # quotes_set.add_flow_step("map3", MyMap, mul=2, after='map')
    # quotes_set.add_flow_step("map4", MyMap, mul=4, after='map')
    quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["5h", "600s"], "1s")

    df = quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )
    print(quotes_set.to_yaml())
    # print(df)

    df = client.ingest(quotes_set, quotes)
    print(quotes_set.to_yaml())
    print(df)

    # stocks_set = FeatureSet("stocks", entities=[Entity("ticker", ValueType.STRING)])
    # client.ingest(stocks_set, stocks, infer_schema=True, with_stats=True)



def test_realtime_query():
    client = fs.store_client(data_prefix="v3io:///users/admin/fs")
    client._default_ingest_targets = [TargetTypes.parquet, TargetTypes.nosql]
    # client.nosql_path_prefix = 'users/admin/fs'

    fset = client.get_feature_set("stock-quotes")
    print(fset.to_yaml())

    features = ["stock-quotes:bid", "stock-quotes:ask", "stock-quotes:xx"]
    svc = client.get_online_feature_service(features)
    svc.get([{"ticker": "GOOG"}])
