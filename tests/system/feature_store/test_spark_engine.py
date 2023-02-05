# Copyright 2018 Iguazio
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
import sys
import uuid
from datetime import datetime

import fsspec
import pandas as pd
import pytest
import v3iofs
from pandas._testing import assert_frame_equal
from storey import EmitEveryEvent

import mlrun
import mlrun.feature_store as fstore
from mlrun import code_to_function, store_manager
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import (
    CSVTarget,
    NoSqlTarget,
    ParquetTarget,
    RedisNoSqlTarget,
)
from mlrun.feature_store import FeatureSet
from mlrun.feature_store.steps import (
    DateExtractor,
    DropFeatures,
    MapValues,
    OneHotEncoder,
)
from mlrun.features import Entity
from mlrun.model import DataTarget
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.data_sample import stocks
from tests.system.feature_store.expected_stats import expected_stats


def read_and_assert(csv_path_spark, csv_path_storey):
    read_back_df_spark = None
    file_system = fsspec.filesystem("v3io")
    for file_entry in file_system.ls(csv_path_spark):
        filepath = file_entry["name"]
        if not filepath.endswith("/_SUCCESS"):
            read_back_df_spark = pd.read_csv(f"v3io://{filepath}")
            break
    assert read_back_df_spark is not None

    read_back_df_storey = None
    for file_entry in file_system.ls(csv_path_storey):
        filepath = file_entry["name"]
        read_back_df_storey = pd.read_csv(f"v3io://{filepath}")
        break
    assert read_back_df_storey is not None

    read_back_df_storey = read_back_df_storey.dropna(axis=1, how="all")
    read_back_df_spark = read_back_df_spark.dropna(axis=1, how="all")

    assert read_back_df_spark.sort_index(axis=1).equals(
        read_back_df_storey.sort_index(axis=1)
    )


@TestMLRunSystem.skip_test_if_env_not_configured
# Marked as enterprise because of v3io mount and remote spark
@pytest.mark.enterprise
class TestFeatureStoreSparkEngine(TestMLRunSystem):
    project_name = "fs-system-spark-engine"
    spark_service = ""
    pq_source = "testdata.parquet"
    pq_target = "testdata_target.parquet"
    csv_source = "testdata.csv"
    spark_image_deployed = (
        False  # Set to True if you want to avoid the image building phase
    )
    test_branch = ""  # For testing specific branch. e.g.: "https://github.com/mlrun/mlrun.git@development"

    @classmethod
    def _init_env_from_file(cls):
        env = cls._get_env_from_file()
        cls.spark_service = env["MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"]

    @classmethod
    def get_local_pq_source_path(cls):
        return os.path.relpath(str(cls.get_assets_path() / cls.pq_source))

    @classmethod
    def get_remote_pq_source_path(cls, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + cls.pq_source
        return path

    def get_remote_pq_target_path(self, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + self.pq_target
        return path

    @classmethod
    def get_local_csv_source_path(cls):
        return os.path.relpath(str(cls.get_assets_path() / cls.csv_source))

    @classmethod
    def get_remote_csv_source_path(cls, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + cls.csv_source
        return path

    @classmethod
    def custom_setup_class(cls):
        from mlrun import get_run_db
        from mlrun.run import new_function
        from mlrun.runtimes import RemoteSparkRuntime

        cls._init_env_from_file()

        store, _ = store_manager.get_or_create_store(cls.get_remote_pq_source_path())
        store.upload(
            cls.get_remote_pq_source_path(without_prefix=True),
            cls.get_local_pq_source_path(),
        )
        store, _ = store_manager.get_or_create_store(cls.get_remote_csv_source_path())
        store.upload(
            cls.get_remote_csv_source_path(without_prefix=True),
            cls.get_local_csv_source_path(),
        )

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

    def test_basic_remote_spark_ingest(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        fstore.ingest(
            measurements,
            source,
            return_df=True,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        assert measurements.status.targets[0].run_id is not None

        stats_df = measurements.get_stats_table()
        expected_stats_df = pd.DataFrame(expected_stats)
        print(f"stats_df: {stats_df.to_json()}")
        print(f"expected_stats_df: {expected_stats_df.to_json()}")
        assert stats_df.equals(expected_stats_df)

    def test_basic_remote_spark_ingest_csv(self):
        key = "patient_id"
        name = "measurements"
        measurements = fstore.FeatureSet(
            name,
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        # Added to test that we can ingest a column named "summary"
        measurements.graph.to(name="rename_column", handler="rename_column")
        source = CSVSource(
            "mycsv",
            path=self.get_remote_csv_source_path(),
        )
        filename = str(
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "spark_ingest_remote_test_code.py"
        )
        func = code_to_function("func", kind="remote-spark", filename=filename)
        run_config = fstore.RunConfig(
            local=False, function=func, handler="ingest_handler"
        )
        fstore.ingest(
            measurements,
            source,
            return_df=True,
            spark_context=self.spark_service,
            run_config=run_config,
        )

        features = [f"{name}.*"]
        vec = fstore.FeatureVector("test-vec", features)

        resp = fstore.get_offline_features(vec, with_indexes=True)
        df = resp.to_dataframe()
        assert type(df["timestamp"][0]).__name__ == "Timestamp"

    def test_error_flow(self):
        df = pd.DataFrame(
            {
                "name": ["Jean", "Jacques", "Pierre"],
                "last_name": ["Dubois", "Dupont", "Lavigne"],
            }
        )

        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity("name")],
            engine="spark",
        )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            fstore.ingest(
                measurements,
                df,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=False),
            )

    def test_ingest_to_csv(self):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fstore.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fstore.ingest(
            measurements,
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")

        read_back_df_spark = None
        file_system = fsspec.filesystem("v3io")
        for file_entry in file_system.ls(csv_path_spark):
            filepath = file_entry["name"]
            if not filepath.endswith("/_SUCCESS"):
                read_back_df_spark = pd.read_csv(f"v3io://{filepath}")
                break
        assert read_back_df_spark is not None

        read_back_df_storey = None
        for file_entry in file_system.ls(csv_path_storey):
            filepath = file_entry["name"]
            read_back_df_storey = pd.read_csv(f"v3io://{filepath}")
            break
        assert read_back_df_storey is not None

        assert read_back_df_spark.sort_index(axis=1).equals(
            read_back_df_storey.sort_index(axis=1)
        )

    @pytest.mark.skipif(
        not mlrun.mlconf.redis.url,
        reason="mlrun.mlconf.redis.url is not set, skipping until testing against real redis",
    )
    def test_ingest_to_redis(self):
        key = "patient_id"
        name = "measurements_spark"

        measurements = fstore.FeatureSet(
            name,
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [RedisNoSqlTarget()]
        measurements.set_targets(targets, with_defaults=False)
        fstore.ingest(
            measurements,
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(False),
            overwrite=True,
        )
        # read the dataframe from the redis back
        vector = fstore.FeatureVector("myvector", features=[f"{name}.*"])
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get([{"patient_id": "305-90-1613"}])
            assert resp == [
                {
                    "bad": 95,
                    "department": "01e9fe31-76de-45f0-9aed-0f94cc97bca0",
                    "room": 2,
                    "hr": 220.0,
                    "hr_is_error": False,
                    "rr": 25,
                    "rr_is_error": False,
                    "spo2": 99,
                    "spo2_is_error": False,
                    "movements": 4.614601941071927,
                    "movements_is_error": False,
                    "turn_count": 0.3582583538239813,
                    "turn_count_is_error": False,
                    "is_in_bed": 1,
                    "is_in_bed_is_error": False,
                }
            ]

    # tests that data is filtered by time in scheduled jobs
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

        source = ParquetSource(
            "myparquet",
            path=path,
            schedule="mock",  # to enable filtering by time
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
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

        fstore.ingest(
            feature_set,
            source,
            run_config=fstore.RunConfig(local=False),
            targets=targets,
            spark_context=self.spark_service,
        )

        features = [f"{name}.*"]
        vec = fstore.FeatureVector("sched_test-vec", features)

        with fstore.get_online_feature_service(vec) as svc:

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

            fstore.ingest(
                feature_set,
                source,
                run_config=fstore.RunConfig(local=False),
                targets=targets,
                spark_context=self.spark_service,
            )

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
        resp = fstore.get_offline_features(vec)
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

        source = ParquetSource("myparquet", path=path)

        data_set = fstore.FeatureSet(
            f"{name}_storey",
            entities=[Entity("first_name"), Entity("last_name")],
            timestamp_key="time",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max"],
            windows="1h",
            period="10m",
        )

        df = fstore.ingest(data_set, source, targets=[])

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

        data_set = fstore.FeatureSet(
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

        fstore.ingest(
            data_set,
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )

        features = [
            f"{name_spark}.*",
        ]

        vector = fstore.FeatureVector("my-vec", features)
        resp = fstore.get_offline_features(
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

        path = "v3io:///bigdata/test_aggregations_emit_every_event.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        df.to_parquet(path=path, filesystem=fsys)

        source = ParquetSource("myparquet", path=path)
        name_spark = f"{name}_spark"

        data_set = fstore.FeatureSet(
            name_spark,
            entities=[Entity("first_name"), Entity("last_name")],
            timestamp_key="time",
            engine="spark",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "count"],
            windows=["2h"],
            period="10m",
            emit_policy=EmitEveryEvent(),
        )

        fstore.ingest(
            data_set,
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )

        print(f"Results:\n{data_set.to_dataframe().sort_values('time').to_string()}\n")
        result_dict = data_set.to_dataframe().sort_values("time").to_dict(orient="list")

        expected_results = df.to_dict(orient="list")
        expected_results.update(
            {
                "bid_sum_2h": [2000, 10, 2012, 26, 34],
                "bid_max_2h": [2000, 10, 2000, 16, 16],
                "bid_count_2h": [1, 1, 2, 2, 3],
            }
        )
        assert result_dict == expected_results

        # Compare Spark-generated results with Storey results (which are always per-event)
        name_storey = f"{name}_storey"

        storey_data_set = fstore.FeatureSet(
            name_storey,
            timestamp_key="time",
            entities=[Entity("first_name"), Entity("last_name")],
        )

        storey_data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "count"],
            windows=["2h"],
            period="10m",
        )
        fstore.ingest(storey_data_set, source)

        storey_df = storey_data_set.to_dataframe().reset_index().sort_values("time")
        print(f"Storey results:\n{storey_df.to_string()}\n")
        storey_result_dict = storey_df.to_dict(orient="list")

        assert storey_result_dict == result_dict

    def test_mix_of_partitioned_and_nonpartitioned_targets(self):
        name = "test_mix_of_partitioned_and_nonpartitioned_targets"

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

        source = ParquetSource(
            "myparquet",
            path=path,
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        partitioned_output_path = "v3io:///bigdata/partitioned/"
        nonpartitioned_output_path = "v3io:///bigdata/nonpartitioned/"
        targets = [
            ParquetTarget(
                name="tar1",
                path=partitioned_output_path,
                partitioned=True,
            ),
            ParquetTarget(
                name="tar2", path=nonpartitioned_output_path, partitioned=False
            ),
        ]

        fstore.ingest(
            feature_set,
            source,
            run_config=fstore.RunConfig(local=False),
            targets=targets,
            spark_context=self.spark_service,
        )

        partitioned_df = pd.read_parquet(partitioned_output_path)
        partitioned_df.set_index(["time"], inplace=True)
        partitioned_df.drop(columns=["year", "month", "day", "hour"], inplace=True)
        nonpartitioned_df = pd.read_parquet(nonpartitioned_output_path)
        nonpartitioned_df.set_index(["time"], inplace=True)

        pd.testing.assert_frame_equal(
            partitioned_df,
            nonpartitioned_df,
            check_index_type=True,
            check_column_type=True,
        )

    def test_write_empty_dataframe_overwrite_false(self):
        name = "test_write_empty_dataframe_overwrite_false"

        path = "v3io:///bigdata/test_write_empty_dataframe_overwrite_false.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        empty_df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                ],
                "first_name": ["moshe"],
                "data": [2000],
            }
        )[0:0]
        empty_df.to_parquet(path=path, filesystem=fsys)

        source = ParquetSource(
            "myparquet",
            path=path,
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        target = ParquetTarget(
            name="pq",
            path="v3io:///bigdata/test_write_empty_dataframe_overwrite_false/",
            partitioned=False,
        )

        fstore.ingest(
            feature_set,
            source,
            run_config=fstore.RunConfig(local=False),
            targets=[
                target,
            ],
            overwrite=False,
            spark_context=self.spark_service,
        )

        # check that no files were written
        with pytest.raises(FileNotFoundError):
            pd.read_parquet(target.get_target_path())

    def test_write_dataframe_overwrite_false(self):
        name = "test_write_dataframe_overwrite_false"

        path = "v3io:///bigdata/test_write_dataframe_overwrite_false.parquet"
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                ],
                "first_name": ["moshe"],
                "data": [2000],
            }
        )
        df.to_parquet(path=path, filesystem=fsys)

        source = ParquetSource(
            "myparquet",
            path=path,
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        target = ParquetTarget(
            name="pq",
            path="v3io:///bigdata/test_write_dataframe_overwrite_false/",
            partitioned=False,
        )

        fstore.ingest(
            feature_set,
            source,
            run_config=fstore.RunConfig(local=False),
            targets=[
                target,
            ],
            overwrite=False,
            spark_context=self.spark_service,
        )

        features = [f"{name}.*"]
        vec = fstore.FeatureVector("test-vec", features)

        resp = fstore.get_offline_features(vec)
        df = resp.to_dataframe()
        assert df.to_dict() == {"data": {0: 2000}}

    @pytest.mark.parametrize(
        "should_succeed, is_parquet, is_partitioned, target_path",
        [
            # spark - csv - fail for single file
            (True, False, None, "v3io:///bigdata/dif-eng/csv"),
            (False, False, None, "v3io:///bigdata/dif-eng/file.csv"),
            # spark - parquet - fail for single file
            (True, True, True, "v3io:///bigdata/dif-eng/pq"),
            (False, True, True, "v3io:///bigdata/dif-eng/file.pq"),
            (True, True, False, "v3io:///bigdata/dif-eng/pq"),
            (False, True, False, "v3io:///bigdata/dif-eng/file.pq"),
        ],
    )
    def test_different_paths_for_ingest_on_spark_engines(
        self, should_succeed, is_parquet, is_partitioned, target_path
    ):
        fset = FeatureSet("fsname", entities=[Entity("ticker")], engine="spark")

        source = (
            "v3io:///bigdata/test_different_paths_for_ingest_on_spark_engines.parquet"
        )
        fsys = fsspec.filesystem(v3iofs.fs.V3ioFS.protocol)
        stocks.to_parquet(path=source, filesystem=fsys)
        source = ParquetSource(
            "myparquet",
            path=source,
        )

        target = (
            ParquetTarget(name="tar", path=target_path, partitioned=is_partitioned)
            if is_parquet
            else CSVTarget(name="tar", path=target_path)
        )

        if should_succeed:
            fstore.ingest(
                fset,
                run_config=fstore.RunConfig(local=False),
                spark_context=self.spark_service,
                source=source,
                targets=[target],
            )

            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.get_filesystem(False)
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fstore.ingest(fset, source=source, targets=[target])

    def test_error_is_properly_propagated(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path="wrong-path.pq")
        with pytest.raises(mlrun.runtimes.utils.RunError):
            fstore.ingest(
                measurements,
                source,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=False),
            )

    # ML-3092
    @pytest.mark.parametrize("timestamp_key", [None, "timestamp"])
    def test_get_offline_features_with_filter_and_indexes(self, timestamp_key):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key=timestamp_key,
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        fstore.ingest(
            measurements,
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        assert measurements.status.targets[0].run_id is not None
        fv_name = "measurements-fv"
        features = [
            "measurements.bad",
            "measurements.department",
        ]
        my_fv = fstore.FeatureVector(
            fv_name,
            features,
        )
        my_fv.spec.with_indexes = True
        my_fv.save()
        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp = fstore.get_offline_features(
            fv_name,
            target=target,
            query="bad>6 and bad<8",
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
        )
        resp_df = resp.to_dataframe()
        target_df = target.as_df()
        source_df = source.to_dataframe()
        expected_df = source_df[source_df["bad"] == 7][["bad", "department"]]
        expected_df.reset_index(drop=True, inplace=True)
        assert resp_df.equals(target_df)
        assert resp_df[["bad", "department"]].equals(expected_df)

    # ML-2802
    @pytest.mark.parametrize("passthrough", [True, False])
    def test_get_offline_features_with_spark_engine(self, passthrough):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
            passthrough=passthrough,
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        fstore.ingest(
            measurements,
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        assert measurements.status.targets[0].run_id is not None

        # assert that online target exist (nosql) and offline target does not (parquet)
        if passthrough:
            assert len(measurements.status.targets) == 1
            assert isinstance(measurements.status.targets["nosql"], DataTarget)

        fv_name = "measurements-fv"
        features = [
            "measurements.bad",
            "measurements.department",
        ]
        my_fv = fstore.FeatureVector(
            fv_name,
            features,
        )
        my_fv.save()
        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp = fstore.get_offline_features(
            fv_name,
            target=target,
            query="bad>6 and bad<8",
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
        )
        resp_df = resp.to_dataframe()
        target_df = target.as_df()
        source_df = source.to_dataframe()
        expected_df = source_df[source_df["bad"] == 7][["bad", "department"]]
        expected_df.reset_index(drop=True, inplace=True)
        assert resp_df.equals(target_df)
        assert resp_df[["bad", "department"]].equals(expected_df)

    def test_ingest_with_steps_drop_features(self):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(DropFeatures(features=["bad"]))
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fstore.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(DropFeatures(features=["bad"]))
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fstore.ingest(
            measurements,
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        read_and_assert(csv_path_spark, csv_path_storey)

    def test_ingest_with_steps_onehot(self):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(OneHotEncoder(mapping={"is_in_bed": [0, 1]}))
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fstore.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(OneHotEncoder(mapping={"is_in_bed": [0, 1]}))
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fstore.ingest(
            measurements,
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        read_and_assert(csv_path_spark, csv_path_storey)

    @pytest.mark.parametrize("with_original_features", [True, False])
    def test_ingest_with_steps_mapval(self, with_original_features):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(
            MapValues(
                mapping={
                    "bad": {"ranges": {"one": [0, 30], "two": [30, "inf"]}},
                    "hr_is_error": {False: "0", True: "1"},
                },
                with_original_features=with_original_features,
            )
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fstore.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(
            MapValues(
                mapping={
                    "bad": {"ranges": {"one": [0, 30], "two": [30, "inf"]}},
                    "hr_is_error": {False: "0", True: "1"},
                },
                with_original_features=with_original_features,
            )
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fstore.ingest(
            measurements,
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        read_and_assert(csv_path_spark, csv_path_storey)

    @pytest.mark.parametrize("timestamp_col", [None, "timestamp"])
    def test_ingest_with_steps_extractor(self, timestamp_col):
        key = "patient_id"
        csv_path_spark = "v3io:///bigdata/test_ingest_to_csv_spark"
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(
            DateExtractor(
                parts=["day_of_year"],
                timestamp_col=timestamp_col,
            )
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fstore.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(
            DateExtractor(
                parts=["day_of_year"],
                timestamp_col=timestamp_col,
            )
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fstore.ingest(
            measurements,
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        read_and_assert(csv_path_spark, csv_path_storey)

    @pytest.mark.parametrize("with_indexes", [True, False])
    @pytest.mark.parametrize("join_type", ["inner", "outer"])
    def test_relation_join(self, join_type, with_indexes):
        """Test 3 option of using get offline feature with relations"""
        departments = pd.DataFrame(
            {
                "d_id": [i for i in range(1, 11, 2)],
                "name": [f"dept{num}" for num in range(1, 11, 2)],
                "manager_id": [i for i in range(10, 15)],
            }
        )

        employees_with_department = pd.DataFrame(
            {
                "id": [num for num in range(100, 600, 100)],
                "name": [f"employee{num}" for num in range(100, 600, 100)],
                "department_id": [1, 1, 2, 6, 11],
            }
        )

        employees_with_class = pd.DataFrame(
            {
                "id": [100, 200, 300],
                "name": [f"employee{num}" for num in range(100, 400, 100)],
                "department_id": [1, 1, 2],
                "class_id": [i for i in range(20, 23)],
            }
        )

        classes = pd.DataFrame(
            {
                "c_id": [i for i in range(20, 30, 2)],
                "name": [f"class{num}" for num in range(20, 30, 2)],
            }
        )

        managers = pd.DataFrame(
            {
                "m_id": [i for i in range(10, 20, 2)],
                "name": [f"manager{num}" for num in range(10, 20, 2)],
            }
        )

        join_employee_department = pd.merge(
            employees_with_department,
            departments,
            how=join_type,
            left_on=["department_id"],
            right_on=["d_id"],
            suffixes=("_employees", "_departments"),
        )

        join_employee_managers = pd.merge(
            join_employee_department,
            managers,
            how=join_type,
            left_on=["manager_id"],
            right_on=["m_id"],
            suffixes=("_manage", "_"),
        )

        join_employee_sets = pd.merge(
            employees_with_department,
            employees_with_class,
            how=join_type,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_employees", "_e_mini"),
        )

        _merge_step = pd.merge(
            join_employee_department,
            employees_with_class,
            how=join_type,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_", "_e_mini"),
        )

        join_all = pd.merge(
            _merge_step,
            classes,
            how=join_type,
            left_on=["class_id"],
            right_on=["c_id"],
            suffixes=("_e_mini", "_cls"),
        )

        col_1 = ["name_employees", "name_departments"]
        col_2 = ["name_employees", "name_departments", "name"]
        col_3 = ["name_employees", "name_e_mini"]
        col_4 = ["name_employees", "name_departments", "name_e_mini", "name_cls"]
        if with_indexes:
            join_employee_department.set_index(["id", "d_id"], drop=True, inplace=True)
            join_employee_managers.set_index(
                ["id", "d_id", "m_id"], drop=True, inplace=True
            )
            join_employee_sets.set_index(["id"], drop=True, inplace=True)
            join_all.set_index(["id", "d_id", "c_id"], drop=True, inplace=True)

        join_employee_department = (
            join_employee_department[col_1]
            .rename(
                columns={"name_departments": "n2", "name_employees": "n"},
            )
            .sort_values(by="n", ignore_index=True)
        )

        join_employee_managers = (
            join_employee_managers[col_2]
            .rename(
                columns={
                    "name_departments": "n2",
                    "name_employees": "n",
                    "name": "man_name",
                },
            )
            .sort_values(by="n", ignore_index=True)
        )

        join_employee_sets = (
            join_employee_sets[col_3]
            .rename(
                columns={"name_employees": "n", "name_e_mini": "mini_name"},
            )
            .sort_values(by="n", ignore_index=True)
        )

        join_all = (
            join_all[col_4]
            .rename(
                columns={
                    "name_employees": "n",
                    "name_departments": "n2",
                    "name_e_mini": "mini_name",
                },
            )
            .sort_values(by="n", ignore_index=True)
        )

        # relations according to departments_set relations
        managers_set_entity = fstore.Entity("m_id")
        managers_set = fstore.FeatureSet(
            "managers",
            entities=[managers_set_entity],
        )
        managers_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(managers_set, managers)

        classes_set_entity = fstore.Entity("c_id")
        classes_set = fstore.FeatureSet(
            "classes",
            entities=[classes_set_entity],
        )
        managers_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(classes_set, classes)

        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments",
            entities=[departments_set_entity],
            relations={"manager_id": managers_set_entity},
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(departments_set, departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(employees_set, employees_with_department)

        mini_employees_set = fstore.FeatureSet(
            "mini-employees",
            entities=[employees_set_entity],
            relations={
                "department_id": departments_set_entity,
                "class_id": classes_set_entity,
            },
        )
        mini_employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(mini_employees_set, employees_with_class)

        features = ["employees.name"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            join_type=join_type,
            order_by="name",
        )
        if with_indexes:
            expected = pd.DataFrame(employees_with_department, columns=["id", "name"]).set_index('id', drop=True)
            assert_frame_equal(expected, resp.to_dataframe())
        else:
            assert_frame_equal(
                pd.DataFrame(employees_with_department, columns=["name"]),
                resp.to_dataframe(),
            )
        features = ["employees.name as n", "departments.name as n2"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()

        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp_1 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            join_type=join_type,
            order_by="n",
        )
        assert_frame_equal(join_employee_department, resp_1.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "managers.name as man_name",
        ]

        vector = fstore.FeatureVector(
            "man-vec", features, description="Employees feature vector"
        )
        vector.save()

        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp_2 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            join_type=join_type,
            order_by=["n"],
        )
        assert_frame_equal(join_employee_managers, resp_2.to_dataframe())

        features = ["employees.name as n", "mini-employees.name as mini_name"]

        vector = fstore.FeatureVector(
            "mini-emp-vec", features, description="Employees feature vector"
        )
        vector.save()

        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp_3 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            join_type=join_type,
            order_by="name",
        )
        assert_frame_equal(join_employee_sets, resp_3.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "mini-employees.name as mini_name",
            "classes.name as name_cls",
        ]

        vector = fstore.FeatureVector(
            "four-vec", features, description="Employees feature vector"
        )
        vector.save()

        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp_4 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            join_type=join_type,
            order_by="n",
        )
        assert_frame_equal(join_all, resp_4.to_dataframe())

    @pytest.mark.parametrize("with_indexes", [False, True])
    def test_relation_asof_join(self, with_indexes):
        """Test 3 option of using get offline feature with relations"""
        departments = pd.DataFrame(
            {
                "d_id": [i for i in range(1, 11, 2)],
                "name": [f"dept{num}" for num in range(1, 11, 2)],
                "manager_id": [i for i in range(10, 15)],
                "time": [pd.Timestamp(f"2023-01-01 0{i}:00:00") for i in range(5)],
            }
        )

        employees_with_department = pd.DataFrame(
            {
                "id": [num for num in range(100, 600, 100)],
                "name": [f"employee{num}" for num in range(100, 600, 100)],
                "department_id": [1, 1, 2, 6, 11],
                "time": [pd.Timestamp(f"2023-01-01 0{i}:00:00") for i in range(5)],
            }
        )

        join_employee_department = pd.merge_asof(
            employees_with_department,
            departments,
            left_on=["time"],
            right_on=["time"],
            left_by=["department_id"],
            right_by=["d_id"],
            suffixes=("_employees", "_departments"),
        )

        col_1 = ["name_employees", "name_departments"]
        if with_indexes:
            col_1 = ["name_employees", "name_departments", "time"]
            join_employee_department.set_index(["id", "d_id"], drop=True)

        join_employee_department = join_employee_department[col_1].rename(
            columns={"name_departments": "n2", "name_employees": "n"},
        )

        # relations according to departments_set relations
        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments", entities=[departments_set_entity], timestamp_key="time"
        )
        departments_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(departments_set, departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
            timestamp_key="time",
        )
        employees_set.set_targets(targets=["parquet"], with_defaults=False)
        fstore.ingest(employees_set, employees_with_department)

        features = ["employees.name as n", "departments.name as n2"]

        vector = fstore.FeatureVector(
            "employees-vec", features, description="Employees feature vector"
        )
        vector.save()
        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        resp_1 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=False, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
        )

        assert_frame_equal(join_employee_department, resp_1.to_dataframe())
