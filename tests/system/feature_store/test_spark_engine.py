import os

import pytest

import mlrun.feature_store as fs
from mlrun import store_manager
from mlrun.datastore.sources import ParquetSource
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
# Marked as enterprise because of v3io mount and remote spark
@pytest.mark.enterprise
class TestFeatureStoreSparkEngine(TestMLRunSystem):
    project_name = "fs-system-spark-engine"
    spark_service = ""
    pq_source = "testdata.parquet"
    spark_image_deployed = (
        False  # Set to True if you want to avoid the image building phase
    )
    test_branch = ""  # For testing specific branche. e.g.: "https://github.com/mlrun/mlrun.git@development"

    @classmethod
    def _init_env_from_file(cls):
        env = cls._get_env_from_file()
        cls.spark_service = env["MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"]

    def get_local_pq_source_path(self):
        return os.path.relpath(str(self.assets_path / self.pq_source))

    def get_remote_pq_source_path(self, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + self.pq_source
        return path

    def custom_setup(self):
        from mlrun import get_run_db
        from mlrun.run import new_function
        from mlrun.runtimes import RemoteSparkRuntime

        if not self.spark_image_deployed:
            self._init_env_from_file()

            store, _ = store_manager.get_or_create_store(
                self.get_remote_pq_source_path()
            )
            store.upload(
                self.get_remote_pq_source_path(without_prefix=True),
                self.get_local_pq_source_path(),
            )
            if not self.test_branch:
                RemoteSparkRuntime.deploy_default_image()
            else:
                sj = new_function(
                    kind="remote-spark", name="remote-spark-default-image-deploy-temp"
                )

                sj.spec.build.image = RemoteSparkRuntime.default_image
                sj.with_spark_service(spark_service="dummy-spark")
                sj.spec.build.commands = ["pip install git+" + self.test_branch]
                sj.deploy(with_mlrun=False)
                get_run_db().delete_function(name=sj.metadata.name)
            self.spark_image_deployed = True

    def test_basic_remote_spark_ingest(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        fs.ingest(
            measurements,
            source,
            return_df=True,
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )
