from data_sample import quotes, trades, stocks
import mlrun.featurestore as fs


def test_get_offline():
    client = fs.store_client(data_prefix="./store")

    # add feature set with time (for time travel) and record its stats
    quotes_set = fs.FeatureSet("stock-quotes")
    quotes_set.infer_from_df(
        quotes, entity_columns=["ticker"], with_stats=True, timestamp_key="time"
    )

    # save ingest data and print the FeatureSet spec
    client.ingest(quotes_set, quotes)
    print(quotes_set.to_yaml())

    # add feature set without time column (stock ticker metadata)
    stocks_set = fs.FeatureSet("stocks").infer_from_df(
        stocks, entity_columns=["ticker"]
    )
    client.ingest(stocks_set, stocks)

    features = ["stock-quotes:bid", "stock-quotes:ask@mycol", "stocks:*"]
    resp = client.get_offline_features(
        features, entity_rows=trades, entity_timestamp_column="time"
    )
    print(resp.to_dataframe())

    service = client.get_online_feature_service(features)
    service.get(trades)
