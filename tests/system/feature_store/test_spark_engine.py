import os
from datetime import datetime
from time import sleep

import fsspec
import pandas as pd
import pytest
import v3iofs

import mlrun
import mlrun.feature_store as fs
from mlrun import store_manager
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import NoSqlTarget, ParquetTarget
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

    def test_error_flow(self):
        df = pd.DataFrame(
            {
                "name": ["Jean", "Jacques", "Pierre"],
                "last_name": ["Dubois", "Dupont", "Lavigne"],
            }
        )

        measurements = fs.FeatureSet(
            "measurements", entities=[fs.Entity("name")], engine="spark",
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(
                measurements,
                df,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fs.RunConfig(local=False),
            )

    @pytest.mark.parametrize("partitioned", [True, False])
    def test_schedule_on_filtered_by_time(self, partitioned):
        name = f"sched-time-{str(partitioned)}"

        now = datetime.now()

        path = "v3io:///bigdata/bla.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                    pd.Timestamp("2021-01-10 11:00:00"),
                ],
                "first_name": ["moshe", "yosi"],
                "data": [2000, 10],
            }
        ).to_parquet(path=path, filesystem=fsys)

        cron_trigger = "*/2 * * * *"

        source = ParquetSource(
            "myparquet", path=path, time_field="time", schedule=cron_trigger
        )

        feature_set = fs.FeatureSet(
            name=name,
            entities=[fs.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        if partitioned:
            targets = [
                NoSqlTarget(),
                ParquetTarget(
                    name="tar1",
                    path="v3io:///bigdata/fs1/",
                    partitioned=True,
                    partition_cols=["time"],
                ),
            ]
        else:
            targets = [
                ParquetTarget(
                    name="tar2", path="v3io:///bigdata/fs2/", partitioned=False
                ),
                NoSqlTarget(),
            ]

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False),
            targets=targets,
            spark_context=self.spark_service,
        )
        # ingest starts every second minute and it takes ~90 seconds to finish.
        if (now.minute % 2) == 0:
            sleep(60 - now.second + 60 + 90)
        else:
            sleep(60 - now.second + 90)

        features = [f"{name}.*"]
        vec = fs.FeatureVector("sched_test-vec", features)

        svc = fs.get_online_feature_service(vec)

        resp = svc.get([{"first_name": "yosi"}, {"first_name": "moshe"}])
        assert resp[0]["data"] == 10
        assert resp[1]["data"] == 2000

        pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 12:00:00"),
                    pd.Timestamp("2021-01-10 13:00:00"),
                    now + pd.Timedelta(minutes=10),
                    pd.Timestamp("2021-01-09 13:00:00"),
                ],
                "first_name": ["moshe", "dina", "katya", "uri"],
                "data": [50, 10, 25, 30],
            }
        ).to_parquet(path=path)

        sleep(120)
        resp = svc.get(
            [
                {"first_name": "yosi"},
                {"first_name": "moshe"},
                {"first_name": "katya"},
                {"first_name": "dina"},
                {"first_name": "uri"},
            ]
        )
        assert resp[0]["data"] == 10
        assert resp[1]["data"] == 50
        assert resp[2] is None
        assert resp[3]["data"] == 10
        assert resp[4] is None

        svc.close()

        # check offline
        resp = fs.get_offline_features(vec)
        assert len(resp.to_dataframe() == 4)
        assert "uri" not in resp.to_dataframe() and "katya" not in resp.to_dataframe()
