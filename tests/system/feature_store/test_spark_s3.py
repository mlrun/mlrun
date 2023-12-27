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
import tempfile

import pandas as pd
import pytest

import mlrun.feature_store as fstore
from mlrun import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
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
class TestFeatureStoreSparkEngine(TestMLRunSystem):
    """
    This suite tests feature store functionality with the remote spark runtime (spark service). It does not test spark
    operator. Make sure that, in env.yml, MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE is set to the name of a spark service
    that exists on the remote system, or alternative set spark_service (below) to that name.

    To run the tests against code other than mlrun/mlrun@development, set test_branch below.

    After any tests have already run at least once, you may want to set spark_image_deployed=True (below) to avoid
    rebuilding the image on subsequent runs, as it takes several minutes.

    It is also possible to run most tests in this suite locally if you have pyspark installed. To run locally, set
    run_local=True. This can be very useful for debugging.

    To use s3 instead of v3io as a remote location, set use_s3_as_remote = True
    """

    ds_profile_name = "spark_s3_ds_profile"

    project_name = "fs-system-spark-engine"
    spark_service = None
    pq_source = "testdata.parquet"
    pq_target = "testdata_target"
    run_local = True
    spark_image_deployed = (
        True  # Set to True if you want to avoid the image building phase
    )
    test_branch = ""  # For testing specific branch. e.g.: "https://github.com/mlrun/mlrun.git@development"

    src_path = None
    target_path = None

    def _print_full_df(self, df: pd.DataFrame, df_name: str, passthrough: bool) -> None:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            self._logger.info(f"{df_name}-passthrough_{passthrough}:")
            self._logger.info(df)

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
            if not cls.test_branch:
                RemoteSparkRuntime.deploy_default_image()
            else:
                sj = new_function(
                    kind="remote-spark", name="remote-spark-default-image-deploy-temp"
                )

                sj.spec.build.image = RemoteSparkRuntime.default_image
                sj.with_spark_service(spark_service="dummy-spark")
                sj.spec.build.commands = ["pip install git+" + cls.test_branch]
                sj.deploy(with_mlrun=False)
                get_run_db().delete_function(name=sj.metadata.name)
            cls.spark_image_deployed = True

    @staticmethod
    def is_path_spark_metadata(path):
        return path.endswith("/_SUCCESS") or path.endswith(".crc")

    def setup_method(self, method):
        super().setup_method(method)
        ds_profile = DatastoreProfileS3(
            name=self.ds_profile_name,
            access_key=self.env["AWS_ACCESS_KEY_ID"],
            secret_key=self.env["AWS_SECRET_ACCESS_KEY"],
        )

        if not self.run_local:
            self.spark_service = self.env["MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"]

        register_temporary_client_datastore_profile(ds_profile)
        bucket = self.env["AWS_BUCKET_NAME"]
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

        if self.run_local:
            self._tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self, method):
        super().teardown_method(method)

    def test_basic_remote_spark_ingest(self):
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
