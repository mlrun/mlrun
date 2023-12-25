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

import pandas as pd
import pytest

import mlrun.feature_store as fstore
from mlrun import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileAzureBlob,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import (
    ParquetTarget,
)
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.expected_stats import expected_stats


@TestMLRunSystem.skip_test_if_env_not_configured
# Marked as enterprise because of v3io mount and remote spark
@pytest.mark.enterprise
class TestFeatureStoreAzureSparkEngine(TestMLRunSystem):
    """
    This suite tests feature store functionality with the remote spark runtime (spark service). It does not test spark
    operator. Make sure that, in env.yml, MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE is set to the name of a spark service
    that exists on the remote system, or alternative set spark_service (below) to that name.

    To run the tests against code other than mlrun/mlrun@development, set test_branch below.

    After any tests have already run at least once, you may want to set spark_image_deployed=True (below) to avoid
    rebuilding the image on subsequent runs, as it takes several minutes.

    It is also possible to run most tests in this suite locally if you have pyspark installed. To run locally, set
    run_local=True. This can be very useful for debugging.

    """

    ds_profile_name = "spark_azure_ds_profile"

    project_name = "fs-system-spark-engine"
    spark_service = None
    pq_source = "testdata.parquet"
    pq_target = "testdata_target"
    run_local = False
    spark_image_deployed = (
        False  # Set to True if you want to avoid the image building phase
    )
    test_branch = ""  # For testing specific branch. e.g.: "https://github.com/mlrun/mlrun.git@development"
    src_path = None
    target_path = None

    @classmethod
    def custom_setup_class(cls):
        cls.env = cls._get_env_from_file()
        if not cls.run_local:
            cls._setup_remote_run()

    @classmethod
    def _setup_remote_run(cls):
        from mlrun import get_run_db
        from mlrun.run import new_function
        from mlrun.runtimes import RemoteSparkRuntime

        if not cls.spark_image_deployed:
            sj = new_function(
                kind="remote-spark", name="remote-spark-azure-default-image-deploy-temp"
            )
            if not cls.test_branch:
                sj.spec.build.commands = ["pip install azure-storage-blob"]
            else:
                sj.spec.build.commands = [
                    "pip install azure-storage-blob git+" + cls.test_branch
                ]
            sj.spec.build.image = RemoteSparkRuntime.default_image
            sj.with_spark_service(spark_service="dummy-spark-azure")
            sj.deploy(with_mlrun=False)
            get_run_db().delete_function(name=sj.metadata.name)
            cls.spark_image_deployed = True

    def setup_method(self, method):
        super().setup_method(method)
        ds_profile = DatastoreProfileAzureBlob(
            name=self.ds_profile_name,
            connection_string=self.env["AZURE_STORAGE_CONNECTION_STRING"],
        )

        if not self.run_local:
            self.spark_service = self.env["MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"]

        register_temporary_client_datastore_profile(ds_profile)
        bucket = self.env["AZURE_CONTAINER"]
        self.src_path = f"ds://{ds_profile.name}/{bucket}/bigdata/{self.pq_source}"
        self.target_path = (
            f"ds://{ds_profile.name}/{bucket}/bigdata/{self.project_name}/spark_output"
        )
        store, _ = store_manager.get_or_create_store(f"ds://{ds_profile.name}/{bucket}")
        store.upload(
            f"/bigdata/{self.pq_source}",
            os.path.relpath(str(self.get_assets_path() / self.pq_source)),
        )

        store.get(f"/bigdata/{self.pq_source}")
        self.project.register_datastore_profile(ds_profile)

    def test_basic_remote_spark_ingest_azure(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.set_targets(
            [ParquetTarget(path=self.target_path)], with_defaults=False
        )
        source = ParquetSource("myparquet", path=self.src_path)
        fstore.ingest(
            measurements,
            source,
            return_df=True,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        assert measurements.status.targets[0].run_id is not None

        stats_df = measurements.get_stats_table()
        expected_stats_df = pd.DataFrame(expected_stats)
        print(f"stats_df: {stats_df.to_json()}")
        print(f"expected_stats_df: {expected_stats_df.to_json()}")
        assert stats_df.equals(expected_stats_df)
