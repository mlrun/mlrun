# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import pathlib
import typing

import pandas as pd
import pytest

import mlrun.feature_store as fstore
from mlrun.datastore.sources import BigQuerySource
from mlrun.datastore.targets import ParquetTarget
from tests.system.base import TestMLRunSystem

CREDENTIALS_ENV = "MLRUN_SYSTEM_TESTS_GOOGLE_BIG_QUERY_CREDENTIALS_JSON_PATH"
CREDENTIALS_JSON_DEFAULT_PATH = (
    TestMLRunSystem.root_path / "tests" / "system" / "google-big-query-credentials.json"
)


def resolve_google_credentials_json_path() -> typing.Optional[pathlib.Path]:
    default_path = pathlib.Path(CREDENTIALS_JSON_DEFAULT_PATH)
    if os.getenv(CREDENTIALS_ENV):
        return pathlib.Path(os.getenv(CREDENTIALS_ENV))
    elif default_path.exists():
        return default_path
    return None


def are_google_credentials_not_set() -> bool:
    credentials_path = resolve_google_credentials_json_path()
    return not credentials_path


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    are_google_credentials_not_set(),
    reason=f"Environment variable {CREDENTIALS_ENV} is not defined, and credentials file not in default path"
    f" {CREDENTIALS_JSON_DEFAULT_PATH}, skipping...",
)
@pytest.mark.enterprise
class TestFeatureStoreGoogleBigQuery(TestMLRunSystem):
    project_name = "fs-system-test-google-big-query"
    max_results = 100

    @classmethod
    def ingest_and_assert(cls, name: str, source: BigQuerySource):
        credentials_path = resolve_google_credentials_json_path()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

        targets = [
            ParquetTarget(
                f"{name}_target",
                path=f"v3io:///projects/big-query/FeatureStore/big-query/{name}/parquet/sets/",
                time_partitioning_granularity="day",
            )
        ]
        feature_set_name = "taxis"
        feature_set = fstore.FeatureSet(
            feature_set_name,
            entities=[fstore.Entity("unique_key")],
            timestamp_key="trip_start_timestamp",
            engine="pandas",
        )
        ingest_df = fstore.ingest(feature_set, source, targets)
        assert ingest_df is not None
        assert len(ingest_df) == cls.max_results
        assert ingest_df.dtypes["pickup_latitude"] == "float64"
        assert ingest_df.dtypes["trip_seconds"] == pd.Int64Dtype()

    @pytest.mark.parametrize("chunksize", [None, 30])
    def test_big_query_source_query(self, chunksize):
        query_string = f"select *\nfrom `bigquery-public-data.chicago_taxi_trips.taxi_trips`\nlimit {self.max_results}"
        source = BigQuerySource(
            "BigQuerySource",
            query=query_string,
            materialization_dataset="chicago_taxi_trips",
            chunksize=chunksize,
        )
        self.ingest_and_assert("query", source)

    @pytest.mark.parametrize("chunksize", [None, 50])
    def test_big_query_source_table(self, chunksize):
        source = BigQuerySource(
            "BigQuerySource",
            table="bigquery-public-data.chicago_taxi_trips.taxi_trips",
            max_results_for_table=self.max_results,
            materialization_dataset="chicago_taxi_trips",
            chunksize=chunksize,
        )
        self.ingest_and_assert("table_c", source)
