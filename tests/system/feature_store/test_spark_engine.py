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
from time import sleep

import fsspec
import pandas as pd
import pytest
import v3iofs
from storey import EmitEveryEvent

import mlrun
import mlrun.feature_store as fs
from mlrun import code_to_function, store_manager
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import CSVTarget, NoSqlTarget, ParquetTarget
from mlrun.feature_store import FeatureSet
from mlrun.features import Entity
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.data_sample import stocks


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

    def get_local_pq_source_path(self):
        return os.path.relpath(str(self.assets_path / self.pq_source))

    def get_remote_pq_source_path(self, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + self.pq_source
        return path

    def get_remote_pq_target_path(self, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + self.pq_target
        return path

    def get_local_csv_source_path(self):
        return os.path.relpath(str(self.assets_path / self.csv_source))

    def get_remote_csv_source_path(self, without_prefix=False):
        path = "v3io://"
        if without_prefix:
            path = ""
        path += "/bigdata/" + self.csv_source
        return path

    def custom_setup(self):
        from mlrun import get_run_db
        from mlrun.run import new_function
        from mlrun.runtimes import RemoteSparkRuntime

        self._init_env_from_file()

        store, _ = store_manager.get_or_create_store(self.get_remote_pq_source_path())
        store.upload(
            self.get_remote_pq_source_path(without_prefix=True),
            self.get_local_pq_source_path(),
        )
        store, _ = store_manager.get_or_create_store(self.get_remote_csv_source_path())
        store.upload(
            self.get_remote_csv_source_path(without_prefix=True),
            self.get_local_csv_source_path(),
        )

        if not self.spark_image_deployed:
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
        assert measurements.status.targets[0].run_id is not None

    def test_basic_remote_spark_ingest_csv(self):
        key = "patient_id"
        name = "measurements"
        measurements = fs.FeatureSet(
            name,
            entities=[fs.Entity(key)],
            engine="spark",
        )
        # Added to test that we can ingest a column named "summary"
        measurements.graph.to(name="rename_column", handler="rename_column")
        source = CSVSource(
            "mycsv", path=self.get_remote_csv_source_path(), time_field="timestamp"
        )
        filename = str(
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "spark_ingest_remote_test_code.py"
        )
        func = code_to_function("func", kind="remote-spark", filename=filename)
        run_config = fs.RunConfig(local=False, function=func, handler="ingest_handler")
        fs.ingest(
            measurements,
            source,
            return_df=True,
            spark_context=self.spark_service,
            run_config=run_config,
        )

        features = [f"{name}.*"]
        vec = fs.FeatureVector("test-vec", features)

        resp = fs.get_offline_features(vec)
        df = resp.to_dataframe()
        assert type(df["timestamp"][0]).__name__ == "Timestamp"

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
        csv_path_storey = "v3io:///bigdata/test_ingest_to_csv_storey.csv"

        measurements = fs.FeatureSet(
            "measurements_spark",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        fs.ingest(
            measurements,
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fs.FeatureSet(
            "measurements_storey",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
        )
        source = ParquetSource("myparquet", path=self.get_remote_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        fs.ingest(
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

        cron_trigger = "*/3 * * * *"

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
        # ingest starts every third minute and it can take ~150 seconds to finish.
        time_till_next_run = 180 - now.second - 60 * (now.minute % 3)
        sleep(time_till_next_run + 150)

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

            sleep(180)
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
            timestamp_key="time",
            entities=[Entity("first_name"), Entity("last_name")],
        )

        storey_data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "count"],
            windows=["2h"],
            period="10m",
        )
        fs.ingest(storey_data_set, source)

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
            time_field="time",
        )

        feature_set = fs.FeatureSet(
            name=name,
            entities=[fs.Entity("first_name")],
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

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False),
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
            time_field="time",
        )

        feature_set = fs.FeatureSet(
            name=name,
            entities=[fs.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        target = ParquetTarget(
            name="pq",
            path="v3io:///bigdata/test_write_empty_dataframe_overwrite_false/",
            partitioned=False,
        )

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False),
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
            time_field="time",
        )

        feature_set = fs.FeatureSet(
            name=name,
            entities=[fs.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        target = ParquetTarget(
            name="pq",
            path="v3io:///bigdata/test_write_dataframe_overwrite_false/",
            partitioned=False,
        )

        fs.ingest(
            feature_set,
            source,
            run_config=fs.RunConfig(local=False),
            targets=[
                target,
            ],
            overwrite=False,
            spark_context=self.spark_service,
        )

        features = [f"{name}.*"]
        vec = fs.FeatureVector("test-vec", features)

        resp = fs.get_offline_features(vec)
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
            fs.ingest(
                fset,
                run_config=fs.RunConfig(local=False),
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
                fs.ingest(fset, source=source, targets=[target])

    def test_error_is_properly_propagated(self):
        key = "patient_id"
        measurements = fs.FeatureSet(
            "measurements",
            entities=[fs.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path="wrong-path.pq")
        with pytest.raises(mlrun.runtimes.utils.RunError):
            fs.ingest(
                measurements,
                source,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fs.RunConfig(local=False),
            )

    def test_get_offline_features_with_filter(self):
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
            spark_context=self.spark_service,
            run_config=fs.RunConfig(local=False),
        )
        assert measurements.status.targets[0].run_id is not None

        fv_name = "measurements-fv"
        features = [
            "measurements.bad",
            "measurements.department",
        ]

        my_fv = fs.FeatureVector(
            fv_name,
            features,
            description="my feature vector",
        )
        my_fv.save()
        target = ParquetTarget("mytarget", path=self.get_remote_pq_target_path())
        fs.get_offline_features(
            fv_name,
            target=target,
            query="bad>6 and bad<8",
            engine="spark",
            run_config=fs.RunConfig(local=False),
        )
        df_res = target.as_df()
        df = source.to_dataframe()
        expected_df = df[df["bad"] == 7][["bad", "department"]]
        expected_df.reset_index(drop=True, inplace=True)

        assert df_res.equals(expected_df)
