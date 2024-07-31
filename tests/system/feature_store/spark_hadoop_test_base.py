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
from enum import Enum

import pandas as pd
import pytest

import mlrun.feature_store as fstore
from mlrun import store_manager
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import (
    ParquetTarget,
)
from mlrun.runtimes import RemoteSparkRuntime
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.expected_stats import expected_stats


class Deployment(Enum):
    Local = 1
    Remote = 2


@pytest.mark.enterprise
class SparkHadoopTestBase(TestMLRunSystem):
    """
    This class serves as a support tool for testing Spark with file systems based on Hadoop,
    both as a source and a target. It is not designed for testing Spark operators.

    To conduct a test with a specific filesystem, you should derive your test class from `SparkHadoopTestBase`.
    In your test class, override the `custom_setup_class()` method following the example provided below.
    Then, in your test, ensure to call the `do_test()` method of this class.

    Here's an illustration of how a derived test class might look:
    class TestFeatureStoreSparkEngine(SparkHadoopTestBase):
        @classmethod
        def custom_setup_class(cls):
            cls.configure_namespace("mytest")
            cls.configure_image_deployment(
                Deployment.Remote,
                "azure-storage-blob git+https://github.com/mlrun/mlrun.git@development",
            )
        def test_basic_remote_spark_ingest_ds_basic(self):
            bucket = 'my_bucket'
            ds_profile = DatastoreProfileBasic(name=self.ds_profile_name)
            register_temporary_client_datastore_profile(ds_profile)
            self.project.register_datastore_profile(ds_profile)
            self.ds_upload_src(ds_profile, bucket)
            self.do_test(
                self.ds_src_path(ds_profile, bucket),
                self.ds_target_path(ds_profile, bucket),
            )

    It is also possible to run the tests locally if you have pyspark installed. To run locally, call
    configure_image_deployment(deployment_type=Deployment.Local). This can be very useful for debugging.
    """

    @classmethod
    def configure_namespace(cls, test_suite_name="hadoop"):
        cls.project_name = "fs-system-spark-engineaaaa-" + test_suite_name
        cls.ds_profile_name = "ds_profile_spark_" + test_suite_name
        cls.remote_function_name = (
            "remote-spark-default-image-deploy-temp-" + test_suite_name
        )
        cls.spark_service_name = cls._get_env_from_file().get(
            "MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"
        )
        cls.pq_source = "testdata.parquet"
        cls.pq_target = "testdata_target"

    @classmethod
    def configure_image_deployment(
        cls,
        deployment_type=Deployment.Remote,
        additional_pip_packages=None,
        spark_image_deployed=False,
    ):
        cls.deployment_type = deployment_type
        if cls.deployment_type == Deployment.Local:
            cls.spark_image_deployed = True
        else:
            cls.spark_image_deployed = spark_image_deployed

        if not cls.spark_image_deployed:
            from mlrun import get_run_db
            from mlrun.run import new_function

            sj = new_function(kind="remote-spark", name=cls.remote_function_name)
            if additional_pip_packages:
                sj.spec.build.commands = [f"pip install {additional_pip_packages}"]
            sj.spec.build.image = RemoteSparkRuntime.default_image
            sj.with_spark_service(spark_service=cls.spark_service_name)
            sj.deploy(with_mlrun=False)
            get_run_db().delete_function(name=sj.metadata.name)
            cls.spark_image_deployed = True

    def ds_src_path(self, ds_profile, bucket):
        return f"ds://{ds_profile.name}/{bucket}/bigdata/{self.pq_source}"

    def ds_target_path(self, ds_profile, bucket):
        return (
            f"ds://{ds_profile.name}/{bucket}/bigdata/{self.project_name}/spark_output"
        )

    def ds_upload_src(self, ds_profile, bucket):
        store, _, _ = store_manager.get_or_create_store(
            f"ds://{ds_profile.name}/{bucket}"
        )
        store.upload(
            f"/bigdata/{self.pq_source}",
            os.path.relpath(str(self.get_assets_path() / self.pq_source)),
        )
        store.get(f"/bigdata/{self.pq_source}")

    def do_test(self, src_path, target_path):
        if self.deployment_type == Deployment.Remote:
            spark_service = self.spark_service_name
        else:
            spark_service = None

        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.set_targets([ParquetTarget(path=target_path)], with_defaults=False)
        source = ParquetSource("myparquet", path=src_path)
        measurements.ingest(
            source,
            return_df=True,
            spark_context=spark_service,
            run_config=fstore.RunConfig(
                local=(self.deployment_type == Deployment.Local)
            ),
        )
        assert measurements.status.targets[0].run_id is not None

        stats_df = measurements.get_stats_table()
        expected_stats_df = pd.DataFrame(expected_stats)
        print(f"stats_df: {stats_df.to_json()}")
        print(f"expected_stats_df: {expected_stats_df.to_json()}")
        assert stats_df.equals(expected_stats_df)
