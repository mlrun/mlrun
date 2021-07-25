import dask.dataframe as dd
import mlrun.datastore


def test_http_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"
    )
    data_item.as_df()

def test_load_object_into_dask_dataframe():
    data_item = mlrun.datastore.store_manager.object(
        "s3://iguazio/data/market-palce/aggregate/metrics.pq"
    )
    ddf = data_item.as_df(df_module = dd)
    assert hasattr(ddf, "dask")
