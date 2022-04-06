import os
import uuid
from datetime import datetime
from time import sleep

import fsspec
import pandas as pd
import pytest
import v3iofs
from storey import EmitEveryEvent

import mlrun
import mlrun.feature_store as fs
from mlrun import store_manager
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import CSVTarget, NoSqlTarget, ParquetTarget
from mlrun.features import Entity
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
    test_branch = "https://github.com/gtopper/mlrun.git@ML-1919"

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

        self._init_env_from_file()

        if not self.spark_image_deployed:

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
            "measurements",
            entities=[fs.Entity("name")],
            engine="spark",
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fs.ingest(
                measurements,
                df,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fs.RunConfig(local=False),
            )

    def test_ingest_to_csv(self):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey"

        measurements = fs.FeatureSet(
            "measurements_spark",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(path=csv_path_spark)]
        fs.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )

        measurements = fs.FeatureSet(
            "measurements_storey",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(path=csv_path_storey)]
        fs.ingest(
            measurements,
            source,
            targets,
        )

        file_system = fsspec.filesystem("v3io")
        spark_output_files = file_system.ls(csv_path_spark)
        assert len(spark_output_files) == 2
        assert "_SUCCESS" in spark_output_files
        csv_filename = [
            filename for filename in spark_output_files if filename != "_SUCCESS"
        ][0]
        pd.read_csv(f"{csv_path_spark}/{csv_filename}")

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

        with fs.get_online_feature_service(vec) as svc:

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

        # check offline
        resp = fs.get_offline_features(vec)
        assert len(resp.to_dataframe() == 4)
        assert "uri" not in resp.to_dataframe() and "katya" not in resp.to_dataframe()

    def test_aggregations(self):
        name = f"measurements_{uuid.uuid4()}"

        test_base_time = datetime.fromisoformat("2020-07-21T21:40:00+00:00")

        df = pd.DataFrame(
            {
                "time": [
                    test_base_time,
                    test_base_time + pd.Timedelta(minutes=1),
                    test_base_time + pd.Timedelta(minutes=2),
                    test_base_time + pd.Timedelta(minutes=3),
                    test_base_time + pd.Timedelta(minutes=4),
                ],
                "first_name": ["moshe", "yosi", "yosi", "moshe", "yosi"],
                "last_name": ["cohen", "levi", "levi", "cohen", "levi"],
                "bid": [2000, 10, 11, 12, 16],
                "mood": ["bad", "good", "bad", "good", "good"],
            }
        )

        path = "v3io:///bigdata/test_aggregations.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        df.to_parquet(path=path, filesystem=fsys)

        source = ParquetSource("myparquet", path=path, time_field="time")

        data_set = fs.FeatureSet(
            f"{name}_storey",
            entities=[Entity("first_name"), Entity("last_name")],
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        df = fs.ingest(data_set, source, targets=[])

        assert df.to_dict() == {
            "mood": {("moshe", "cohen"): "good", ("yosi", "levi"): "good"},
            "bid": {("moshe", "cohen"): 12, ("yosi", "levi"): 16},
            "bid_sum_1h": {("moshe", "cohen"): 2012, ("yosi", "levi"): 37},
            "bid_max_1h": {("moshe", "cohen"): 2000, ("yosi", "levi"): 16},
            "time": {
                ("moshe", "cohen"): pd.Timestamp("2020-07-21 21:43:00Z"),
                ("yosi", "levi"): pd.Timestamp("2020-07-21 21:44:00Z"),
            },
        }

        name_spark = f"{name}_spark"

        data_set = fs.FeatureSet(
            name_spark,
            entities=[Entity("first_name"), Entity("last_name")],
            engine="spark",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        fs.ingest(
            data_set,
            source,
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )

        features = [
            f"{name_spark}.*",
        ]

        vector = fs.FeatureVector("my-vec", features)
        resp = fs.get_offline_features(
            vector, entity_timestamp_column="time", with_indexes=True
        )
        assert resp.to_dataframe().to_dict() == {
            "mood": {("moshe", "cohen"): "good", ("yosi", "levi"): "good"},
            "bid": {("moshe", "cohen"): 12, ("yosi", "levi"): 16},
            "bid_sum_1h": {("moshe", "cohen"): 2012, ("yosi", "levi"): 37},
            "bid_max_1h": {("moshe", "cohen"): 2000, ("yosi", "levi"): 16},
            "time": {
                ("moshe", "cohen"): pd.Timestamp("2020-07-21 22:00:00"),
                ("yosi", "levi"): pd.Timestamp("2020-07-21 22:30:00"),
            },
            "time_window": {("moshe", "cohen"): "1h", ("yosi", "levi"): "1h"},
        }

    def test_aggregations_emit_every_event(self):
        name = f"measurements_{uuid.uuid4()}"
        test_base_time = datetime.fromisoformat("2020-07-21T21:40:00")

        df = pd.DataFrame(
            {
                "time": [
                    test_base_time,
                    test_base_time + pd.Timedelta(minutes=20),
                    test_base_time + pd.Timedelta(minutes=60),
                    test_base_time + pd.Timedelta(minutes=79),
                    test_base_time + pd.Timedelta(minutes=81),
                ],
                "first_name": ["Moshe", "Yossi", "Moshe", "Yossi", "Yossi"],
                "last_name": ["Cohen", "Levi", "Cohen", "Levi", "Levi"],
                "bid": [2000, 10, 12, 16, 8],
            }
        )

        path = "v3io:///bigdata/test_aggregations.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        df.to_parquet(path=path, filesystem=fsys)

        source = ParquetSource("myparquet", path=path, time_field="time")
        name_spark = f"{name}_spark"

        data_set = fs.FeatureSet(
            name_spark,
            entities=[Entity("first_name"), Entity("last_name")],
            engine="spark",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "count"],
            windows=["1h", "2h"],
            period="10m",
            emit_policy=EmitEveryEvent(),
        )

        fs.ingest(
            data_set,
            source,
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )

        print(f"Results:\n{data_set.to_dataframe().sort_values('time').to_string()}\n")
        result_dict = data_set.to_dataframe().sort_values("time").to_dict(orient="list")

        expected_results = df.to_dict(orient="list")
        expected_results.update(
            {
                "bid_sum_1h": [2000, 10, 12, 26, 24],
                "bid_max_1h": [2000, 10, 12, 16, 16],
                "bid_count_1h": [1, 1, 1, 2, 2],
                "bid_sum_2h": [2000, 10, 2012, 26, 34],
                "bid_max_2h": [2000, 10, 2000, 16, 16],
                "bid_count_2h": [1, 1, 2, 2, 3],
            }
        )
        assert result_dict == expected_results

        # Compare Spark-generated results with Storey results (which are always per-event)
        name_storey = f"{name}_storey"

        storey_data_set = fs.FeatureSet(
            name_storey,
            entities=[Entity("first_name"), Entity("last_name")],
        )

        storey_data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "count"],
            windows=["1h", "2h"],
            period="10m",
        )
        fs.ingest(storey_data_set, source)

        storey_df = storey_data_set.to_dataframe().reset_index().sort_values("time")
        print(f"Storey results:\n{storey_df.to_string()}\n")
        storey_result_dict = storey_df.to_dict(orient="list")

        assert storey_result_dict == result_dict
