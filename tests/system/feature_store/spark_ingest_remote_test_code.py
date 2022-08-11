from mlrun.feature_store.api import ingest


def ingest_handler(context):
    ingest(mlrun_context=context)


def rename_column(df):
    df = df.withColumnRenamed("department", "summary")
    return df
