import os

import pandas as pd
import pytest

import mlrun.feature_store as fs
from mlrun.datastore.sources import BigQuerySource
from mlrun.datastore.targets import ParquetTarget
from tests.system.base import TestMLRunSystem

CREDENTIALS_ENV = "MLRUN_SYSTEM_TESTS_GOOGLE_BIG_QUERY_CREDENTIALS_JSON"


def _are_google_credentials_set() -> bool:
    return not os.getenv(CREDENTIALS_ENV)


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    _are_google_credentials_set(),
    reason=f"Environment variable {CREDENTIALS_ENV} is not defined, skipping...",
)
@pytest.mark.enterprise
class TestFeatureStoreGoogleBigQuery(TestMLRunSystem):
    project_name = "fs-system-test-google-big-query"

    def test_big_query_source_query(self):
        max_results = 100
        query_string = f"select *\nfrom `bigquery-public-data.chicago_taxi_trips.taxi_trips`\nlimit {max_results}"
        source = BigQuerySource(
            "BigQuerySource",
            query=query_string,
            materialization_dataset="chicago_taxi_trips",
        )
        self._test_big_query_source("query", source, max_results)

    def test_big_query_source_query_with_chunk_size(self):
        max_results = 100
        query_string = f"select *\nfrom `bigquery-public-data.chicago_taxi_trips.taxi_trips`\nlimit {max_results * 2}"
        source = BigQuerySource(
            "BigQuerySource",
            query=query_string,
            materialization_dataset="chicago_taxi_trips",
            chunksize=max_results,
        )
        self._test_big_query_source("query_c", source, max_results)

    def test_big_query_source_table(self):
        max_results = 100
        source = BigQuerySource(
            "BigQuerySource",
            table="bigquery-public-data.chicago_taxi_trips.taxi_trips",
            max_results_for_table=max_results,
            materialization_dataset="chicago_taxi_trips",
        )
        self._test_big_query_source("table", source, max_results)

    def test_big_query_source_table_with_chunk_size(self):
        max_results = 100
        source = BigQuerySource(
            "BigQuerySource",
            table="bigquery-public-data.chicago_taxi_trips.taxi_trips",
            max_results_for_table=max_results * 2,
            materialization_dataset="chicago_taxi_trips",
            chunksize=max_results,
        )
        self._test_big_query_source("table_c", source, max_results)

    @staticmethod
    def _test_big_query_source(name: str, source: BigQuerySource, max_results: int):
        gbq_credentials_json = str(os.environ.get(CREDENTIALS_ENV))
        with open("google_application_credentials.txt", "w") as fh:
            fh.write(gbq_credentials_json)
        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"
        ] = "google_application_credentials.txt"

        targets = [
            ParquetTarget(
                f"{name}_target",
                path=f"v3io:///projects/big-query/FeatureStore/big-query/{name}/parquet/sets/",
                time_partitioning_granularity="day",
            )
        ]
        feature_set_name = "taxis"
        feature_set = fs.FeatureSet(
            feature_set_name,
            entities=[fs.Entity("unique_key")],
            timestamp_key="trip_start_timestamp",
            engine="pandas",
        )
        ingest_df = fs.ingest(feature_set, source, targets, return_df=False)
        assert ingest_df is not None
        assert len(ingest_df) == max_results
        assert ingest_df.dtypes["pickup_latitude"] == "float64"
        assert ingest_df.dtypes["trip_seconds"] == pd.Int64Dtype()
