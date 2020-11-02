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


def test_ingestion():
    client = fs.store_client(data_prefix="v3io:///users/admin/fs")
    client._default_ingest_targets = [TargetTypes.parquet, TargetTypes.nosql]

    # add feature set without time column (stock ticker metadata)
    #stocks_set = fs.FeatureSet("stocks")
    #resp = stocks_set.infer_from_df(stocks, entity_columns=["ticker"])
    #resp = client.ingest(stocks_set, stocks)
    #print(resp)
    #return


    quotes_set = FeatureSet("stock-quotes")
    quotes_set.add_flow_step("map", "storey.Map", mul=3)
    quotes_set.add_flow_step("map2", "MyMap", mul=3, after='map')
    quotes_set.add_flow_step("map3", "MyMap", mul=3, after='map')
    quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["5h", "600s"], "1s")

    df = quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )

    with open('df.html', 'w') as fp:
        fp.write(df.to_html())

    client.ingest(quotes_set, quotes)
    print(quotes_set.to_yaml())


def test_realtime_query():
    client = fs.store_client(data_prefix="v3io:///users/admin/fs")
    client._default_ingest_targets = [TargetTypes.parquet, TargetTypes.nosql]

    features = ["stock-quotes:*"]
    svc = client.get_online_feature_service(features)
    resp = svc.get([{"ticker": "GOOG"}])
    print(resp)
