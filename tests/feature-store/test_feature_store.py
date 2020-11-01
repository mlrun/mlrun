import asyncio

from data_sample import quotes, trades, stocks
from storey import Flow
from storey.dtypes import _termination_obj
import mlrun.featurestore as fs
from mlrun.featurestore import FeatureSet, Entity
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

    features = ["stock-quotes:bid", "stock-quotes:ask@mycol", "stocks:*"]
    resp = client.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    print(resp.to_dataframe())

    # service = client.get_online_feature_service(features)
    # service.get(trades)


def my_fn(event):
    event["xx"] = event["bid"] * 2
    return event


class MapClass(Flow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_async = asyncio.iscoroutinefunction(self.do)
        self._filter = False

    def filter(self):
        # used in the .do() code to signal filtering
        self._filter = True

    def do(self, event):
        raise NotImplementedError()

    async def _call(self, event):
        res = self.do(event)
        if self._is_async:
            res = await res
        return res

    async def _do(self, event):
        if event is _termination_obj:
            return await self._do_downstream(_termination_obj)
        else:
            element = self._get_safe_event_or_body(event)
            fn_result = await self._call(element)
            if not self._filter:
                mapped_event = self._user_fn_output_to_event(event, fn_result)
                await self._do_downstream(mapped_event)
            else:
                self._filter = False  # clear the flag for future runs


class MyMap(MapClass):
    def __init__(self, mul=1, **kwargs):
        super().__init__(**kwargs)
        self._mul = mul

    def do(self, event):
        event["xx"] = event["bid"] * self._mul
        return event


def test_storey():
    client = fs.store_client(data_prefix="./store")

    quotes_set = FeatureSet(
        "stock-quotes",
        entities=[Entity("ticker", ValueType.STRING)],
        timestamp_key="time",
    )
    quotes_set.add_flow_step("map", MyMap, mul=3)
    # quotes_set.add_flow_step("map3", MyMap, mul=2, after='map')
    # quotes_set.add_flow_step("map4", MyMap, mul=4, after='map')
    quotes_set.add_aggregation("asks", "ask", ["sum", "max"], ["5s", "20s"], "1s")
    df = quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )
    print(quotes_set.to_yaml())
    # print(df)

    df = client.ingest(quotes_set, quotes)
    print(quotes_set.to_yaml())
    print(df)
