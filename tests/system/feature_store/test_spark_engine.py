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
import sys
import tempfile
import uuid
from datetime import datetime

import fsspec
import pandas as pd
import pytest
import requests
import v3iofs
from pandas._testing import assert_frame_equal
from storey import EmitEveryEvent

import mlrun
import mlrun.datastore.utils
import mlrun.feature_store as fstore
from mlrun import code_to_function, store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileHdfs,
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
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
from mlrun.runtimes.utils import RunError
from mlrun.utils.helpers import to_parquet
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.data_sample import stocks
from tests.system.feature_store.expected_stats import expected_stats
from tests.system.feature_store.utils import sort_df


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

    ds_profile = None
    project_name = "fs-system-spark-engine"
    spark_service = ""
    pq_source = "testdata.parquet"
    pq_target = "testdata_target"
    csv_source = "testdata.csv"
    run_local = False
    use_s3_as_remote = False
    spark_image_deployed = (
        False  # Set to True if you want to avoid the image building phase
    )
    test_branch = ""  # For testing specific branch. e.g.: "https://github.com/mlrun/mlrun.git@development"

    @classmethod
    def _init_env_from_file(cls):
        env = cls._get_env_from_file()
        if cls.run_local:
            cls.spark_service = None
        else:
            cls.spark_service = env["MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE"]

    @classmethod
    def get_local_pq_source_path(cls):
        return os.path.relpath(str(cls.get_assets_path() / cls.pq_source))

    @classmethod
    def get_remote_path_prefix(cls, without_prefix):
        if cls.use_s3_as_remote:
            cls.ds_profile = DatastoreProfileS3(
                name="s3ds_profile",
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            )
            register_temporary_client_datastore_profile(cls.ds_profile)
            bucket = os.environ["AWS_BUCKET_NAME"]
            path = f"ds://{cls.ds_profile.name}/{bucket}"
            if without_prefix:
                path = f"{bucket}"
        else:
            path = "v3io://"
            if without_prefix:
                path = ""
        return path

    @classmethod
    def get_remote_pq_source_path(cls, without_prefix=False):
        path = cls.get_remote_path_prefix(without_prefix) + "/bigdata/" + cls.pq_source
        return path

    @classmethod
    def get_pq_source_path(cls):
        if cls.run_local:
            return cls.get_local_pq_source_path()
        else:
            return cls.get_remote_pq_source_path()

    def _print_full_df(self, df: pd.DataFrame, df_name: str, passthrough: bool) -> None:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            self._logger.info(f"{df_name}-passthrough_{passthrough}:")
            self._logger.info(df)

    @classmethod
    def get_local_csv_source_path(cls):
        return os.path.relpath(str(cls.get_assets_path() / cls.csv_source))

    @classmethod
    def get_remote_csv_source_path(cls, without_prefix=False):
        path = cls.get_remote_path_prefix(without_prefix) + "/bigdata/" + cls.csv_source
        return path

    @classmethod
    def get_csv_source_path(cls):
        if cls.run_local:
            return cls.get_local_csv_source_path()
        else:
            return cls.get_remote_csv_source_path()

    @classmethod
    def custom_setup_class(cls):
        cls._init_env_from_file()

        if not cls.run_local:
            cls._setup_remote_run()

    @classmethod
    def _setup_remote_run(cls):
        from mlrun import get_run_db
        from mlrun.run import new_function
        from mlrun.runtimes import RemoteSparkRuntime

        store, _, _ = store_manager.get_or_create_store(cls.get_remote_pq_source_path())
        store.upload(
            cls.get_remote_pq_source_path(without_prefix=True),
            cls.get_local_pq_source_path(),
        )
        store, _, _ = store_manager.get_or_create_store(
            cls.get_remote_csv_source_path()
        )
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

    @staticmethod
    def is_path_spark_metadata(path):
        return path.endswith("/_SUCCESS") or path.endswith(".crc")

    @classmethod
    def read_parquet_and_assert(cls, out_path_spark, out_path_storey):
        read_back_df_spark = None
        file_system = fsspec.filesystem("file" if cls.run_local else "v3io")
        for file_entry in file_system.ls(out_path_spark):
            filepath = file_entry if cls.run_local else f'v3io://{file_entry["name"]}'
            if not cls.is_path_spark_metadata(filepath):
                read_back_df_spark = pd.read_parquet(filepath)
                break
        assert read_back_df_spark is not None

        read_back_df_storey = None
        for file_entry in file_system.ls(out_path_storey):
            filepath = file_entry if cls.run_local else f'v3io://{file_entry["name"]}'
            read_back_df_storey = pd.read_parquet(filepath)
            break
        assert read_back_df_storey is not None

        read_back_df_storey = read_back_df_storey.dropna(axis=1, how="all")
        read_back_df_spark = read_back_df_spark.dropna(axis=1, how="all")

        # spark does not support indexes, so we need to reset the storey result to match it
        read_back_df_storey.reset_index(inplace=True)

        pd.testing.assert_frame_equal(
            read_back_df_spark,
            read_back_df_storey,
            check_categorical=False,
            check_like=True,
            check_dtype=False,
        )

    @classmethod
    def read_csv(cls, csv_path: str) -> pd.DataFrame:
        file_system = fsspec.filesystem("file" if cls.run_local else "v3io")
        if file_system.isdir(csv_path):
            for file_entry in file_system.ls(csv_path):
                filepath = (
                    file_entry if cls.run_local else f'v3io://{file_entry["name"]}'
                )
                if not cls.is_path_spark_metadata(filepath):
                    return pd.read_csv(filepath)
        else:
            return pd.read_csv(csv_path)
        raise AssertionError(f"No files found in {csv_path}")

    @staticmethod
    def read_csv_and_assert(csv_path_spark, csv_path_storey):
        read_back_df_spark = TestFeatureStoreSparkEngine.read_csv(
            csv_path=csv_path_spark
        )
        read_back_df_storey = TestFeatureStoreSparkEngine.read_csv(
            csv_path=csv_path_storey
        )

        read_back_df_storey = read_back_df_storey.dropna(axis=1, how="all")
        read_back_df_spark = read_back_df_spark.dropna(axis=1, how="all")

        pd.testing.assert_frame_equal(
            read_back_df_storey,
            read_back_df_spark,
            check_categorical=False,
            check_like=True,
        )

    def setup_method(self, method):
        super().setup_method(method)
        if self.run_local:
            self._tmpdir = tempfile.TemporaryDirectory()
        if self.ds_profile:
            self.project.register_datastore_profile(self.ds_profile)

    def teardown_method(self, method):
        super().teardown_method(method)
        if self.run_local:
            self._tmpdir.cleanup()

    def output_dir(self, url=True):
        if self.run_local:
            prefix = "file://" if url else ""
            base_dir = f"{prefix}{self._tmpdir.name}"
        else:
            base_dir = f"v3io:///projects/{self.project_name}"
        result = f"{base_dir}/spark-tests-output"
        if self.run_local:
            os.makedirs(result, exist_ok=True)
        return result

    hdfs_output_dir = "ds://my-hdfs/test/system-test-output"

    @staticmethod
    def get_test_name():
        return (
            os.environ.get("PYTEST_CURRENT_TEST")
            .split(":")[-1]
            .split(" ")[0]
            .replace("[", "__")
            .replace("]", "")
        )

    def get_test_output_subdir_path(self, url=True):
        return f"{self.output_dir(url=url)}/{self.get_test_name()}"

    def set_targets(self, feature_set, also_in_remote=False):
        dir_name = self.get_test_name()
        if self.run_local or also_in_remote:
            target_path = f"{self.output_dir(url=False)}/{dir_name}"
            feature_set.set_targets(
                [ParquetTarget(path=target_path)], with_defaults=False
            )

    def test_basic_remote_spark_ingest(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        self.set_targets(measurements)
        measurements.ingest(
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

    def test_special_columns_missing(self):
        key = "patient_id"
        entity_fset = fstore.FeatureSet(
            "entity_fset",
            entities=[fstore.Entity(key.upper())],
            engine="spark",
        )
        timestamp_fset = fstore.FeatureSet(
            "timestamp_fset",
            timestamp_key="TIMESTAMP",
            engine="spark",
        )

        label_fset = fstore.FeatureSet(
            "label_fset",
            label_column="BAD",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        self.set_targets(entity_fset, also_in_remote=True)
        self.set_targets(timestamp_fset, also_in_remote=True)
        self.set_targets(label_fset, also_in_remote=True)
        error_type = (
            mlrun.errors.MLRunInvalidArgumentError if self.run_local else RunError
        )

        with pytest.raises(
            error_type,
            match="There are missing entities from dataframe during ingestion.",
        ):
            entity_fset.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

        with pytest.raises(
            error_type,
            match="timestamp_key is missing from dataframe during ingestion.",
        ):
            timestamp_fset.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

        with pytest.raises(
            error_type,
            match="label_column is missing from dataframe during ingestion.",
        ):
            label_fset.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

    @pytest.mark.parametrize("passthrough", [True, False])
    def test_parquet_filters(self, passthrough):
        parquet_source_path = self.get_pq_source_path()
        source_file_name = "testdata_with_none.parquet"
        parquet_source_path = parquet_source_path.replace(
            self.pq_source, source_file_name
        )
        if not self.run_local:
            df = pd.read_parquet(
                self.get_local_pq_source_path().replace(
                    self.pq_source, source_file_name
                )
            )
            df.to_parquet(parquet_source_path)

        filters = [("department", "in", ["01e9fe31-76de-45f0-9aed-0f94cc97bca0", None])]
        filtered_df = pd.read_parquet(parquet_source_path, filters=filters)
        base_path = self.get_test_output_subdir_path()
        parquet_target_path = f"{base_path}_spark"
        parquet_source = ParquetSource(
            "parquet_source",
            path=parquet_source_path,
            additional_filters=filters,
        )
        feature_set = fstore.FeatureSet(
            "parquet-filters-fs",
            entities=[fstore.Entity("patient_id")],
            engine="spark",
            passthrough=passthrough,
        )

        target = ParquetTarget(
            name="department_based_target",
            path=parquet_target_path,
        )
        run_config = fstore.RunConfig(local=self.run_local)
        feature_set.ingest(
            source=parquet_source,
            targets=[target],
            spark_context=self.spark_service,
            run_config=run_config,
        )
        if not passthrough:
            result = sort_df(
                pd.read_parquet(feature_set.get_target_path()), "patient_id"
            )
            expected = sort_df(filtered_df, "patient_id")
            assert_frame_equal(
                result, expected, check_dtype=False, check_categorical=False
            )

        vec = fstore.FeatureVector(
            name="test-fs-vec", features=["parquet-filters-fs.*"]
        )
        vec.save()
        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        kind = None if self.run_local else "remote-spark"
        resp = fstore.get_offline_features(
            feature_vector=vec,
            additional_filters=[
                ("bad", "not in", [38, 100]),
                ("movements", "<", 6),
            ],
            with_indexes=True,
            target=target,
            engine="spark",
            run_config=fstore.RunConfig(local=self.run_local, kind=kind),
            spark_service=self.spark_service,
        )

        result = resp.to_dataframe()
        result.reset_index(drop=False, inplace=True)

        expected = pd.read_parquet(parquet_source_path) if passthrough else filtered_df
        expected = sort_df(
            expected.query("bad not in [38,100] & movements < 6"),
            ["patient_id"],
        )
        result = sort_df(result, ["patient_id"])
        assert_frame_equal(result, expected, check_dtype=False)

        target = ParquetTarget(
            "vector_target", path=f"{self.output_dir()}-get_offline_features_by_vector"
        )
        #  check get_offline_features vector function:
        resp = vec.get_offline_features(
            additional_filters=[
                ("bad", "not in", [38, 100]),
                ("movements", "<", 6),
            ],
            with_indexes=True,
            target=target,
            engine="spark",
            run_config=fstore.RunConfig(local=self.run_local, kind=kind),
            spark_service=self.spark_service,
        )

        result = resp.to_dataframe()
        result.reset_index(drop=False, inplace=True)
        result = sort_df(result, ["patient_id"])
        assert_frame_equal(result, expected, check_dtype=False)

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
            path=self.get_csv_source_path(),
        )
        filename = str(
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "spark_ingest_remote_test_code.py"
        )
        func = code_to_function("func", kind="remote-spark", filename=filename)
        run_config = fstore.RunConfig(
            local=self.run_local, function=func, handler="ingest_handler"
        )
        self.set_targets(measurements)
        measurements.ingest(
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
            measurements.ingest(
                df,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

    def test_ingest_to_csv(self):
        key = "patient_id"
        base_path = self.get_test_output_subdir_path()
        csv_path_spark = f"{base_path}_spark"
        csv_path_storey = f"{base_path}_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        measurements.ingest(
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")

        read_back_df_spark = None
        file_system = fsspec.filesystem("file" if self.run_local else "v3io")
        for file_entry in file_system.ls(csv_path_spark):
            filepath = file_entry if self.run_local else f'v3io://{file_entry["name"]}'
            if not self.is_path_spark_metadata(filepath):
                read_back_df_spark = pd.read_csv(filepath)
                break
        assert read_back_df_spark is not None

        filepath = csv_path_storey if self.run_local else f"v3io://{csv_path_storey}"
        read_back_df_storey = pd.read_csv(filepath)

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
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [RedisNoSqlTarget()]
        measurements.set_targets(targets, with_defaults=False)
        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
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

    @pytest.mark.skipif(
        run_local,
        reason="We don't normally have redis or v3io jars when running locally",
    )
    @pytest.mark.parametrize(
        "target_kind",
        ["Redis", "v3io"] if mlrun.mlconf.redis.url else ["v3io"],
    )
    def test_ingest_multiple_entities(self, target_kind):
        key1 = "patient_id"
        key2 = "bad"
        key3 = "department"
        name = "measurements_spark"

        measurements = fstore.FeatureSet(
            name,
            entities=[fstore.Entity(key1), fstore.Entity(key2), fstore.Entity(key3)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        if target_kind == "Redis":
            targets = [RedisNoSqlTarget()]
        else:
            targets = [NoSqlTarget()]
        measurements.set_targets(targets, with_defaults=False)

        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
            overwrite=True,
        )
        # read the dataframe
        vector = fstore.FeatureVector("myvector", features=[f"{name}.*"])
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get(
                [
                    {
                        "patient_id": "305-90-1613",
                        "bad": 95,
                        "department": "01e9fe31-76de-45f0-9aed-0f94cc97bca0",
                    }
                ]
            )
            assert resp == [
                {
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

    @pytest.mark.skipif(
        not mlrun.mlconf.redis.url,
        reason="mlrun.mlconf.redis.url is not set, skipping until testing against real redis",
    )
    def test_ingest_to_redis_numeric_index(self):
        key = "movements"
        name = "measurements_spark"

        measurements = fstore.FeatureSet(
            name,
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [RedisNoSqlTarget()]
        measurements.set_targets(targets, with_defaults=False)
        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
            overwrite=True,
        )
        # read the dataframe from the redis back
        vector = fstore.FeatureVector("myvector", features=[f"{name}.*"])
        with fstore.get_online_feature_service(vector) as svc:
            resp = svc.get([{"movements": 4.614601941071927}])
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
                    "patient_id": "305-90-1613",
                    "movements_is_error": False,
                    "turn_count": 0.3582583538239813,
                    "turn_count_is_error": False,
                    "is_in_bed": 1,
                    "is_in_bed_is_error": False,
                }
            ]

    # tests that data is filtered by time in scheduled jobs
    @pytest.mark.skipif(run_local, reason="Local scheduling is not supported")
    @pytest.mark.parametrize("partitioned", [True, False])
    def test_schedule_on_filtered_by_time(self, partitioned):
        name = f"sched-time-{str(partitioned)}"

        now = datetime.now()

        path = f"{self.output_dir()}/bla.parquet"
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                    pd.Timestamp("2021-01-10 11:00:00"),
                ],
                "first_name": ["moshe", "yosi"],
                "data": [2000, 10],
            }
        )
        to_parquet(df, path=path, filesystem=fsys)

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
                    path=f"{self.output_dir()}/fs1/",
                    partitioned=True,
                    partition_cols=["time"],
                ),
            ]
        else:
            targets = [
                ParquetTarget(
                    name="tar2", path=f"{self.output_dir()}/fs2/", partitioned=False
                ),
                NoSqlTarget(),
            ]

        feature_set.ingest(
            source,
            run_config=fstore.RunConfig(local=self.run_local),
            targets=targets,
            spark_context=self.spark_service,
        )

        features = [f"{name}.*"]
        vec = fstore.FeatureVector("sched_test-vec", features)

        with fstore.get_online_feature_service(vec) as svc:
            resp = svc.get([{"first_name": "yosi"}, {"first_name": "moshe"}])
            assert resp[0]["data"] == 10
            assert resp[1]["data"] == 2000

            df = pd.DataFrame(
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
            )
            to_parquet(df, path=path)

            feature_set.ingest(
                source,
                run_config=fstore.RunConfig(local=self.run_local),
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

        path = f"{self.output_dir(url=False)}/test_aggregations.parquet"
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        to_parquet(df, path=path, filesystem=fsys)

        source = ParquetSource("myparquet", path=path)

        data_set = fstore.FeatureSet(
            f"{name}_storey",
            entities=[Entity("first_name"), Entity("last_name")],
            timestamp_key="time",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "sqr", "stdvar"],
            windows="1h",
            period="10m",
        )

        df = data_set.ingest(source, targets=[])

        assert df.fillna("NaN-was-here").to_dict("records") == [
            {
                "bid": 2000,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000000,
                "bid_stdvar_1h": "NaN-was-here",
                "bid_sum_1h": 2000,
                "mood": "bad",
                "time": pd.Timestamp("2020-07-21 21:40:00+0000", tz="UTC"),
            },
            {
                "bid": 10,
                "bid_max_1h": 10,
                "bid_sqr_1h": 100,
                "bid_stdvar_1h": "NaN-was-here",
                "bid_sum_1h": 10,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 21:41:00+0000", tz="UTC"),
            },
            {
                "bid": 11,
                "bid_max_1h": 11,
                "bid_sqr_1h": 221,
                "bid_stdvar_1h": 0.5,
                "bid_sum_1h": 21,
                "mood": "bad",
                "time": pd.Timestamp("2020-07-21 21:42:00+0000", tz="UTC"),
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 21:43:00+0000", tz="UTC"),
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 21:44:00+0000", tz="UTC"),
            },
        ]

        assert df.index.equals(
            pd.MultiIndex.from_arrays(
                [
                    ["moshe", "yosi", "yosi", "moshe", "yosi"],
                    ["cohen", "levi", "levi", "cohen", "levi"],
                ],
                names=("first_name", "last_name"),
            )
        )

        name_spark = f"{name}_spark"

        data_set = fstore.FeatureSet(
            name_spark,
            entities=[Entity("first_name"), Entity("last_name")],
            engine="spark",
        )

        data_set.add_aggregation(
            column="bid",
            operations=["sum", "max", "sqr", "stdvar"],
            windows="1h",
            period="10m",
        )
        self.set_targets(data_set)
        data_set.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )

        features = [
            f"{name_spark}.*",
        ]

        vector = fstore.FeatureVector("my-vec", features)
        resp = fstore.get_offline_features(vector, with_indexes=True)

        # We can't count on the order when reading the results back
        result_records = (
            resp.to_dataframe()
            .sort_values(["first_name", "last_name", "time"])
            .to_dict("records")
        )

        assert result_records == [
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 21:50:00"),
                "time_window": "1h",
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:00:00"),
                "time_window": "1h",
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:10:00"),
                "time_window": "1h",
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:20:00"),
                "time_window": "1h",
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:30:00"),
                "time_window": "1h",
            },
            {
                "bid": 12,
                "bid_max_1h": 2000,
                "bid_sqr_1h": 4000144,
                "bid_stdvar_1h": 1976072,
                "bid_sum_1h": 2012,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:40:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 21:50:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:00:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:10:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:20:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:30:00"),
                "time_window": "1h",
            },
            {
                "bid": 16,
                "bid_max_1h": 16,
                "bid_sqr_1h": 477,
                "bid_stdvar_1h": 10.333333333333334,
                "bid_sum_1h": 37,
                "mood": "good",
                "time": pd.Timestamp("2020-07-21 22:40:00"),
                "time_window": "1h",
            },
        ]

        assert df.index.equals(
            pd.MultiIndex.from_arrays(
                [
                    ["moshe", "yosi", "yosi", "moshe", "yosi"],
                    ["cohen", "levi", "levi", "cohen", "levi"],
                ],
                names=("first_name", "last_name"),
            )
        )

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

        path = (
            f"{self.output_dir(url=False)}/test_aggregations_emit_every_event.parquet"
        )
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        to_parquet(df, path=path, filesystem=fsys)

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
            operations=["sum", "max", "count", "sqr", "stdvar"],
            windows=["2h"],
            period="10m",
            emit_policy=EmitEveryEvent(),
        )
        self.set_targets(data_set)
        data_set.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )

        print(f"Results:\n{data_set.to_dataframe().sort_values('time').to_string()}\n")
        result_dict = (
            data_set.to_dataframe()
            .fillna("NaN-was-here")
            .sort_values("time")
            .to_dict(orient="list")
        )

        expected_results = df.to_dict(orient="list")
        expected_results.update(
            {
                "bid_sum_2h": [2000, 10, 2012, 26, 34],
                "bid_max_2h": [2000, 10, 2000, 16, 16],
                "bid_count_2h": [1, 1, 2, 2, 3],
                "bid_sqr_2h": [4000000, 100, 4000144, 356, 420],
                "bid_stdvar_2h": [
                    "NaN-was-here",
                    "NaN-was-here",
                    1976072,
                    18,
                    17.333333333333332,
                ],
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
            operations=["sum", "max", "count", "sqr", "stdvar"],
            windows=["2h"],
            period="10m",
        )
        storey_data_set.ingest(source)

        storey_df = (
            storey_data_set.to_dataframe()
            .fillna("NaN-was-here")
            .reset_index()
            .sort_values("time")
        )
        print(f"Storey results:\n{storey_df.to_string()}\n")
        storey_result_dict = storey_df.to_dict(orient="list")

        assert storey_result_dict == result_dict

    def test_mix_of_partitioned_and_nonpartitioned_targets(self):
        name = "test_mix_of_partitioned_and_nonpartitioned_targets"

        path = f"{self.output_dir(url=False)}/bla.parquet"
        url = f"{self.output_dir()}/bla.parquet"
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                    pd.Timestamp("2021-01-10 11:00:00"),
                ],
                "first_name": ["moshe", "yosi"],
                "data": [2000, 10],
            }
        )
        to_parquet(df, path=path, filesystem=fsys)

        source = ParquetSource(
            "myparquet",
            path=url,
        )

        feature_set = fstore.FeatureSet(
            name=name,
            entities=[fstore.Entity("first_name")],
            timestamp_key="time",
            engine="spark",
        )

        partitioned_output_path = f"{self.output_dir()}/partitioned/"
        nonpartitioned_output_path = f"{self.output_dir()}/nonpartitioned/"
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

        feature_set.ingest(
            source,
            run_config=fstore.RunConfig(local=self.run_local),
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

        path = f"{self.output_dir(url=False)}/test_write_empty_dataframe_overwrite_false.parquet"
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        empty_df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                ],
                "first_name": ["moshe"],
                "data": [2000],
            }
        )[0:0]
        to_parquet(empty_df, path=path, filesystem=fsys)

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
            path=f"{self.output_dir()}/{self.get_test_name()}/",
            partitioned=False,
        )

        feature_set.ingest(
            source,
            run_config=fstore.RunConfig(local=self.run_local),
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

        path = (
            f"{self.output_dir(url=False)}/test_write_dataframe_overwrite_false.parquet"
        )
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        df = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2021-01-10 10:00:00"),
                ],
                "first_name": ["moshe"],
                "data": [2000],
            }
        )
        to_parquet(df, path=path, filesystem=fsys)

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
            path=f"{self.output_dir()}/{self.get_test_name()}/",
            partitioned=False,
        )

        feature_set.ingest(
            source,
            run_config=fstore.RunConfig(local=self.run_local),
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
            (True, False, None, "dif-eng/csv"),
            (False, False, None, "dif-eng/file.csv"),
            # spark - parquet - fail for single file
            (True, True, True, "dif-eng/pq"),
            (False, True, True, "dif-eng/file.pq"),
            (True, True, False, "dif-eng/pq"),
            (False, True, False, "dif-eng/file.pq"),
        ],
    )
    def test_different_paths_for_ingest_on_spark_engines(
        self, should_succeed, is_parquet, is_partitioned, target_path
    ):
        target_path = f"{self.output_dir()}/{target_path}"

        fset = FeatureSet("fsname", entities=[Entity("ticker")], engine="spark")

        source = f"{self.output_dir(url=False)}/test_different_paths_for_ingest_on_spark_engines.parquet"
        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        to_parquet(stocks, path=source, filesystem=fsys)
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
            fset.ingest(
                run_config=fstore.RunConfig(local=self.run_local),
                spark_context=self.spark_service,
                source=source,
                targets=[target],
            )

            if fset.get_target_path().endswith(fset.status.targets[0].run_id + "/"):
                store, _, _ = mlrun.store_manager.get_or_create_store(
                    fset.get_target_path()
                )
                v3io = store.filesystem
                assert v3io.isdir(fset.get_target_path())
        else:
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                fset.ingest(source=source, targets=[target])

    def test_error_is_properly_propagated(self):
        if self.run_local:
            import pyspark.sql.utils

            expected_error = pyspark.sql.utils.AnalysisException
        else:
            expected_error = mlrun.runtimes.utils.RunError

        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path="wrong-path.pq")
        with pytest.raises(expected_error):
            measurements.ingest(
                source,
                return_df=True,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

    # ML-5726
    @pytest.mark.skipif(
        not {"HDFS_HOST", "HDFS_PORT", "HDFS_HTTP_PORT", "HADOOP_USER_NAME"}.issubset(
            os.environ.keys()
        ),
        reason="HDFS host, ports and user name are not defined",
    )
    def test_ingest_and_get_offline_features_with_hdfs(self):
        key = "patient_id"

        datastore_profile = DatastoreProfileHdfs(
            name="my-hdfs",
            host=os.getenv("HDFS_HOST"),
            port=int(os.getenv("HDFS_PORT")),
            http_port=int(os.getenv("HDFS_HTTP_PORT")),
        )
        register_temporary_client_datastore_profile(datastore_profile)
        self.project.register_datastore_profile(datastore_profile)

        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        self.set_targets(measurements)
        run_config = fstore.RunConfig(
            local=self.run_local,
            kind="remote-spark",
            extra_spec={
                "spec": {
                    "env": [
                        {
                            "name": "HADOOP_USER_NAME",
                            "value": os.environ["HADOOP_USER_NAME"],
                        }
                    ]
                }
            },
        )
        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=run_config,
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
        target = ParquetTarget(
            "mytarget", path=f"{self.hdfs_output_dir}-get_offline_features"
        )
        resp = fstore.get_offline_features(
            fv_name,
            target=target,
            query="bad>6 and bad<8",
            engine="spark",
            run_config=run_config,
            spark_service=self.spark_service,
        )
        resp_df = resp.to_dataframe()
        target_df = target.as_df()
        target_df.set_index(key, drop=True, inplace=True)

        source_df = source.to_dataframe()
        source_df.set_index(key, drop=True, inplace=True)
        expected_df = source_df[source_df["bad"] == 7][["bad", "department"]]
        expected_df.reset_index(drop=True, inplace=True)

        pd.testing.assert_frame_equal(resp_df, target_df)

        resp_df.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(resp_df[["bad", "department"]], expected_df)
        target.purge()
        with pytest.raises(FileNotFoundError):
            target.as_df()
        # check that a FileNotFoundError is not raised
        target.purge()

    @pytest.mark.skipif(
        not {"HDFS_HOST", "HDFS_PORT", "HDFS_HTTP_PORT", "HADOOP_USER_NAME"}.issubset(
            os.environ.keys()
        ),
        reason="HDFS host, ports and user name are not defined",
    )
    def test_hdfs_wrong_credentials(self):
        datastore_profile = DatastoreProfileHdfs(
            name="my-hdfs",
            host=os.getenv("HDFS_HOST"),
            port=int(os.getenv("HDFS_PORT")),
            http_port=int(os.getenv("HDFS_HTTP_PORT")),
            user="wrong-user",
        )
        register_temporary_client_datastore_profile(datastore_profile)
        self.project.register_datastore_profile(datastore_profile)
        target = ParquetTarget(
            "mytarget", path=f"{self.hdfs_output_dir}-get_offline_features"
        )
        with pytest.raises(PermissionError):
            target.purge()

    @pytest.mark.skipif(
        not {"HDFS_HOST", "HDFS_PORT", "HDFS_HTTP_PORT", "HADOOP_USER_NAME"}.issubset(
            os.environ.keys()
        ),
        reason="HDFS host, ports and user name are not defined",
    )
    def test_hdfs_empty_host(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("HDFS_HOST")

        datastore_profile = DatastoreProfileHdfs(
            name="my-hdfs",
            port=int(os.environ["HDFS_PORT"]),
            http_port=int(os.environ["HDFS_HTTP_PORT"]),
        )
        register_temporary_client_datastore_profile(datastore_profile)
        self.project.register_datastore_profile(datastore_profile)
        target = ParquetTarget(
            "mytarget", path=f"{self.hdfs_output_dir}-get_offline_features"
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            target.purge()

        monkeypatch.delenv("HDFS_PORT")
        monkeypatch.delenv("HDFS_HTTP_PORT")
        datastore_profile = DatastoreProfileHdfs(
            name="my-hdfs",
        )
        register_temporary_client_datastore_profile(datastore_profile)
        self.project.register_datastore_profile(datastore_profile)
        target = ParquetTarget(
            "mytarget", path=f"{self.hdfs_output_dir}-get_offline_features"
        )
        with pytest.raises(ValueError):
            target.purge()

    @pytest.mark.parametrize("drop_column", ["department", "timestamp"])
    def test_get_offline_features_with_drop_columns(self, drop_column):
        key = "patient_id"
        timestamp_key = "timestamp"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key=timestamp_key,
            engine="spark",
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        self.set_targets(measurements)
        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
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
        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp = fstore.get_offline_features(
            fv_name,
            target=target,
            engine="spark",
            drop_columns=[drop_column],
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            spark_service=self.spark_service,
        )

        resp_df = resp.to_dataframe()
        target_df = target.as_df()
        target_df.set_index(key, drop=True, inplace=True)
        assert resp_df.equals(target_df)

    # ML-2802, ML-3397
    @pytest.mark.parametrize(
        ["target_type", "passthrough"],
        [
            (ParquetTarget, False),
            (ParquetTarget, True),
            (CSVTarget, False),
            (CSVTarget, True),
        ],
    )
    def test_get_offline_features_with_spark_engine(self, passthrough, target_type):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
            passthrough=passthrough,
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        if not passthrough:
            self.set_targets(measurements)
        measurements.ingest(
            source,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        if passthrough:
            assert len(measurements.status.targets) == 0
        elif not self.run_local:
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
        my_fv.save()
        target = target_type(
            "mytarget",
            path=f"{self.output_dir()}-get_offline_features",
        )
        resp = fstore.get_offline_features(
            fv_name,
            target=target,
            query="bad>6 and bad<8",
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
        )
        resp_df = resp.to_dataframe()
        target_df = target.as_df()
        source_df = source.to_dataframe()
        expected_df = source_df[source_df["bad"] == 7][["bad", "department"]]
        expected_df.reset_index(drop=True, inplace=True)
        self._print_full_df(df=resp_df, df_name="resp_df", passthrough=passthrough)
        self._print_full_df(df=target_df, df_name="target_df", passthrough=passthrough)
        self._print_full_df(
            df=expected_df, df_name="expected_df", passthrough=passthrough
        )
        assert resp_df.equals(target_df)
        assert resp_df[["bad", "department"]].equals(expected_df)

    def test_ingest_with_steps_drop_features(self):
        key = "patient_id"
        base_path = self.get_test_output_subdir_path()
        csv_path_spark = f"{base_path}_spark"
        csv_path_storey = f"{base_path}_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(DropFeatures(features=["bad"]))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(DropFeatures(features=["bad"]))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        measurements.ingest(
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        self.read_csv_and_assert(csv_path_spark, csv_path_storey)

    def test_drop_features_validators(self):
        key = "patient_id"
        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(DropFeatures(features=[key]))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        key_as_set = {key}
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=f"^DropFeatures can only drop features, not entities: {key_as_set}$",
        ):
            measurements.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

        measurements = fstore.FeatureSet(
            "measurements_spark",
            label_column="bad",
            engine="spark",
        )
        measurements.graph.to(DropFeatures(features=["bad"]))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="^DropFeatures can not drop label_column: bad$",
        ):
            measurements.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

        measurements = fstore.FeatureSet(
            "measurements_spark",
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(DropFeatures(features=["timestamp"]))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="^DropFeatures can not drop timestamp_key: timestamp$",
        ):
            measurements.ingest(
                source,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

    def test_ingest_with_steps_onehot(self):
        key = "patient_id"
        base_path = self.get_test_output_subdir_path()
        csv_path_spark = f"{base_path}_spark"
        csv_path_storey = f"{base_path}_storey.csv"

        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(OneHotEncoder(mapping={"is_in_bed": [0, 1]}))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        csv_path_spark = measurements.get_target_path(name="csv")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
        )
        measurements.graph.to(OneHotEncoder(mapping={"is_in_bed": [0, 1]}))
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_storey)]
        measurements.ingest(
            source,
            targets,
        )
        csv_path_storey = measurements.get_target_path(name="csv")
        self.read_csv_and_assert(csv_path_spark, csv_path_storey)

    @pytest.mark.parametrize("with_original_features", [True, False])
    def test_ingest_with_steps_mapvalues(self, with_original_features):
        key = "patient_id"
        base_path = self.get_test_output_subdir_path()
        parquet_path_spark = f"{base_path}_spark"
        parquet_path_storey = f"{base_path}_storey"

        measurements = fstore.FeatureSet(
            "measurements_spark",
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
        df = pd.read_parquet(self.get_pq_source_path())
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [ParquetTarget(name="parquet", path=parquet_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        parquet_path_spark = measurements.get_target_path(name="parquet")

        measurements = fstore.FeatureSet(
            "measurements_storey",
            entities=[fstore.Entity(key)],
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
        targets = [
            ParquetTarget(name="parquet", path=parquet_path_storey, partitioned=False)
        ]
        measurements.ingest(
            source,
            targets,
        )
        parquet_path_storey = measurements.get_target_path(name="parquet")
        self.read_parquet_and_assert(parquet_path_spark, parquet_path_storey)

        vector = fstore.FeatureVector(
            "vector",
            ["measurements_spark.*"],
        )

        target = ParquetTarget(
            "get_offline_target", path=f"{self.output_dir()}-get_offline_features"
        )
        resp = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=True,
            run_config=fstore.RunConfig(
                local=self.run_local, kind=None if self.run_local else "remote-spark"
            ),
            engine="spark",
            spark_service=self.spark_service,
        )
        result = resp.to_dataframe()
        result = result.drop(["bad_mapped", "hr_is_error_mapped"], axis=1)
        if not with_original_features:
            columns = ["bad", "hr_is_error"]
            df = df[columns]
        pd.testing.assert_frame_equal(df, result, check_dtype=False)

    def test_mapvalues_with_partial_mapping(self):
        # checks partial mapping -> only part of the values in field are replaced.
        key = "patient_id"
        csv_path_spark = self.get_test_output_subdir_path()
        original_df = pd.read_parquet(self.get_pq_source_path())
        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(
            MapValues(
                mapping={
                    "bad": {17: -1},
                },
                with_original_features=True,
            )
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        csv_path_spark = measurements.get_target_path(name="csv")
        df = self.read_csv(csv_path=csv_path_spark)
        assert not df.empty
        assert not df["bad_mapped"].isna().any()
        assert not df["bad_mapped"].isnull().any()
        assert not (df["bad_mapped"] == 17).any()
        # Note that there are no occurrences of -1 in the "bad" field of the original DataFrame.
        assert len(df[df["bad_mapped"] == -1]) == len(
            original_df[original_df["bad"] == 17]
        )

    def test_mapvalues_with_mixed_types(self):
        key = "patient_id"
        csv_path_spark = self.get_test_output_subdir_path()
        measurements = fstore.FeatureSet(
            "measurements_spark",
            entities=[fstore.Entity(key)],
            timestamp_key="timestamp",
            engine="spark",
        )
        measurements.graph.to(
            MapValues(
                mapping={
                    "hr_is_error": {True: "1"},
                },
                with_original_features=True,
            )
        )
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [CSVTarget(name="csv", path=csv_path_spark)]
        error_type = (
            mlrun.errors.MLRunBadRequestError
            if self.run_local
            else mlrun.runtimes.utils.RunError
        )
        with pytest.raises(
            error_type,
            match="^MapValues - mapping that changes column type must change all values accordingly,"
            " which is not the case for column 'hr_is_error'$",
        ):
            measurements.ingest(
                source,
                targets,
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )

    @pytest.mark.parametrize("timestamp_col", [None, "timestamp"])
    def test_ingest_with_steps_extractor(self, timestamp_col):
        key = "patient_id"
        base_path = self.get_test_output_subdir_path()
        out_path_spark = f"{base_path}_spark"
        out_path_storey = f"{base_path}_storey"

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
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [ParquetTarget(path=out_path_spark)]
        measurements.ingest(
            source,
            targets,
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        out_path_spark = measurements.get_target_path()

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
        source = ParquetSource("myparquet", path=self.get_pq_source_path())
        targets = [ParquetTarget(path=out_path_storey)]
        measurements.ingest(
            source,
            targets,
        )

        out_path_storey = measurements.get_target_path()
        self.read_parquet_and_assert(out_path_spark, out_path_storey)

    @pytest.mark.parametrize("with_indexes", [True, False])
    def test_relation_join(self, with_indexes):
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
            left_on=["department_id"],
            right_on=["d_id"],
            suffixes=("_employees", "_departments"),
        )

        join_employee_managers = pd.merge(
            join_employee_department,
            managers,
            left_on=["manager_id"],
            right_on=["m_id"],
            suffixes=("_manage", "_"),
        )

        join_employee_sets = pd.merge(
            employees_with_department,
            employees_with_class,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_employees", "_e_mini"),
        )

        _merge_step = pd.merge(
            join_employee_department,
            employees_with_class,
            left_on=["id"],
            right_on=["id"],
            suffixes=("_", "_e_mini"),
        )

        join_all = pd.merge(
            _merge_step,
            classes,
            left_on=["class_id"],
            right_on=["c_id"],
            suffixes=("_e_mini", "_cls"),
            how="right",
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
            .sort_values(by="n")
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
            .sort_values(by="n")
        )

        join_employee_sets = (
            join_employee_sets[col_3]
            .rename(
                columns={"name_employees": "n", "name_e_mini": "mini_name"},
            )
            .sort_values(by="n")
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
            .sort_values(by="n")
        )

        # relations according to departments_set relations
        managers_set_entity = fstore.Entity("m_id")
        managers_set = fstore.FeatureSet(
            "managers",
            entities=[managers_set_entity],
        )
        self.set_targets(managers_set, also_in_remote=True)
        managers_set.ingest(managers)

        classes_set_entity = fstore.Entity("c_id")
        classes_set = fstore.FeatureSet(
            "classes",
            entities=[classes_set_entity],
        )
        self.set_targets(classes_set, also_in_remote=True)
        classes_set.ingest(classes)

        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments",
            entities=[departments_set_entity],
            relations={"manager_id": managers_set_entity},
        )
        self.set_targets(departments_set, also_in_remote=True)
        departments_set.ingest(departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
        )
        self.set_targets(employees_set, also_in_remote=True)
        employees_set.ingest(employees_with_department)

        mini_employees_set = fstore.FeatureSet(
            "mini-employees",
            entities=[employees_set_entity],
        )
        self.set_targets(mini_employees_set, also_in_remote=True)
        mini_employees_set.ingest(employees_with_class)

        extra_relations = {
            "mini-employees": {
                "department_id": departments_set_entity,
                "class_id": "c_id",
            }
        }

        features = ["employees.name"]

        vector = fstore.FeatureVector(
            "employees-vec",
            features,
            description="Employees feature vector",
            relations=extra_relations,
        )
        vector.save()

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by="name",
        )
        if with_indexes:
            expected = pd.DataFrame(
                employees_with_department, columns=["id", "name"]
            ).set_index("id", drop=True)
            assert_frame_equal(expected, resp.to_dataframe())
        else:
            assert_frame_equal(
                pd.DataFrame(employees_with_department, columns=["name"]),
                resp.to_dataframe(),
            )
        features = ["employees.name as n", "departments.name as n2"]

        vector = fstore.FeatureVector(
            "employees-vec",
            features,
            description="Employees feature vector",
            relations=extra_relations,
        )
        vector.save()

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp_1 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by="n",
        )
        assert_frame_equal(join_employee_department, resp_1.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "managers.name as man_name",
        ]

        vector = fstore.FeatureVector(
            "man-vec",
            features,
            description="Employees feature vector",
            relations=extra_relations,
        )
        vector.save()

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp_2 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by=["n"],
        )
        assert_frame_equal(join_employee_managers, resp_2.to_dataframe())

        features = ["employees.name as n", "mini-employees.name as mini_name"]

        vector = fstore.FeatureVector(
            "mini-emp-vec",
            features,
            description="Employees feature vector",
            relations=extra_relations,
        )
        vector.save()

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp_3 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by="name",
        )
        assert_frame_equal(join_employee_sets, resp_3.to_dataframe())

        features = [
            "employees.name as n",
            "departments.name as n2",
            "mini-employees.name as mini_name",
            "classes.name as name_cls",
        ]
        join_graph = (
            fstore.JoinGraph(first_feature_set="employees")
            .inner(departments_set)
            .inner("mini-employees")
            .right("classes")
        )

        vector = fstore.FeatureVector(
            "four-vec",
            features,
            join_graph=join_graph,
            description="Employees feature vector",
            relations=extra_relations,
        )
        vector.save()

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp_4 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by="n",
        )
        assert_frame_equal(join_all, resp_4.to_dataframe())

    @pytest.mark.parametrize("with_indexes", [False, True])
    @pytest.mark.parametrize("with_graph", [False, True])
    def test_relation_asof_join(self, with_indexes, with_graph):
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
            col_1 = ["time", "name_employees", "name_departments"]
            join_employee_department.set_index(["id", "d_id"], drop=True, inplace=True)

        join_employee_department = (
            join_employee_department[col_1]
            .rename(
                columns={"name_departments": "n2", "name_employees": "n"},
            )
            .sort_values("n")
        )

        # relations according to departments_set relations
        departments_set_entity = fstore.Entity("d_id")
        departments_set = fstore.FeatureSet(
            "departments", entities=[departments_set_entity], timestamp_key="time"
        )
        self.set_targets(departments_set, also_in_remote=True)
        departments_set.ingest(departments)

        employees_set_entity = fstore.Entity("id")
        employees_set = fstore.FeatureSet(
            "employees",
            entities=[employees_set_entity],
            relations={"department_id": departments_set_entity},
            timestamp_key="time",
        )
        self.set_targets(employees_set, also_in_remote=True)
        employees_set.ingest(employees_with_department)

        features = ["employees.name as n", "departments.name as n2"]
        join_graph = (
            fstore.JoinGraph(first_feature_set="employees").left(
                "departments", asof_join=True
            )
            if with_graph
            else None
        )

        vector = fstore.FeatureVector(
            "employees-vec",
            features,
            description="Employees feature vector",
            join_graph=join_graph,
        )
        vector.save()
        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp_1 = fstore.get_offline_features(
            vector,
            target=target,
            with_indexes=with_indexes,
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            engine="spark",
            spark_service=self.spark_service,
            order_by=["n"],
        )

        assert_frame_equal(
            join_employee_department.sort_index(axis=1),
            resp_1.to_dataframe().sort_index(axis=1),
        )

    @pytest.mark.parametrize("ts_r", ["ts", "ts_r"])
    def test_as_of_join_result(self, ts_r):
        test_base_time = datetime.fromisoformat("2020-07-21T12:00:00+00:00")

        df_left = pd.DataFrame(
            {
                "ent": ["a", "b"],
                "f1": ["a-val", "b-val"],
                "ts": [test_base_time, test_base_time],
            }
        )

        df_right = pd.DataFrame(
            {
                "ent": ["a", "a", "a", "b"],
                ts_r: [
                    test_base_time - pd.Timedelta(minutes=1),
                    test_base_time - pd.Timedelta(minutes=2),
                    test_base_time - pd.Timedelta(minutes=3),
                    test_base_time - pd.Timedelta(minutes=2),
                ],
                "f2": ["newest", "middle", "oldest", "only-value"],
            }
        )

        expected_df = pd.DataFrame(
            {
                "f1": ["a-val", "b-val"],
                "f2": ["newest", "only-value"],
            }
        )
        base_path = self.get_test_output_subdir_path(url=False)
        left_path = f"{base_path}/df_left.parquet"
        right_path = f"{base_path}/df_right.parquet"

        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        fsys.makedirs(base_path, exist_ok=True)
        to_parquet(df_left, path=left_path, filesystem=fsys)
        to_parquet(df_right, path=right_path, filesystem=fsys)

        fset1 = fstore.FeatureSet("fs1-as-of", entities=["ent"], timestamp_key="ts")
        self.set_targets(fset1, also_in_remote=True)
        fset2 = fstore.FeatureSet("fs2-as-of", entities=["ent"], timestamp_key=ts_r)
        self.set_targets(fset2, also_in_remote=True)

        base_url = self.get_test_output_subdir_path()
        left_url = f"{base_url}/df_left.parquet"
        right_url = f"{base_url}/df_right.parquet"

        source_left = ParquetSource("pq1", path=left_url)
        source_right = ParquetSource("pq2", path=right_url)

        fset1.ingest(source_left)
        fset2.ingest(source_right)

        vec_for_spark = fstore.FeatureVector(
            "vec1-spark", ["fs1-as-of.*", "fs2-as-of.*"]
        )
        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )
        resp = fstore.get_offline_features(
            vec_for_spark,
            engine="spark",
            run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
            spark_service=self.spark_service,
            target=target,
        )
        spark_engine_res = resp.to_dataframe().sort_index(axis=1)

        assert_frame_equal(expected_df, spark_engine_res)

    @pytest.mark.parametrize(
        "timestamp_for_filtering",
        [None, "other_ts", "bad_ts", {"fs1": "other_ts"}, {"fs1": "bad_ts"}],
    )
    @pytest.mark.parametrize("passthrough", [True, False])
    def test_time_filter(self, timestamp_for_filtering, passthrough):
        test_base_time = datetime.fromisoformat("2020-07-21T12:00:00")

        df = pd.DataFrame(
            {
                "ent": ["a", "b", "c", "d"],
                "ts_key": [
                    test_base_time - pd.Timedelta(minutes=1),
                    test_base_time - pd.Timedelta(minutes=2),
                    test_base_time - pd.Timedelta(minutes=3),
                    test_base_time - pd.Timedelta(minutes=4),
                ],
                "other_ts": [
                    test_base_time - pd.Timedelta(minutes=4),
                    test_base_time - pd.Timedelta(minutes=3),
                    test_base_time - pd.Timedelta(minutes=2),
                    test_base_time - pd.Timedelta(minutes=1),
                ],
                "val": [1, 2, 3, 4],
            }
        )

        base_path = self.get_test_output_subdir_path(url=False)
        path = f"{base_path}/df_for_filter.parquet"

        fsys = fsspec.filesystem(
            "file" if self.run_local else v3iofs.fs.V3ioFS.protocol
        )
        fsys.makedirs(base_path, exist_ok=True)
        to_parquet(df, path=path, filesystem=fsys)
        source = ParquetSource("pq1", path=path)

        fset1 = fstore.FeatureSet(
            "fs1", entities=["ent"], timestamp_key="ts_key", passthrough=passthrough
        )
        self.set_targets(fset1, also_in_remote=True)

        fset1.ingest(source)

        vec = fstore.FeatureVector("vec1", ["fs1.val"])

        target = ParquetTarget(
            "mytarget", path=f"{self.output_dir()}-get_offline_features"
        )

        if isinstance(timestamp_for_filtering, dict):
            timestamp_for_filtering_str = timestamp_for_filtering["fs1"]
        else:
            timestamp_for_filtering_str = timestamp_for_filtering
        if timestamp_for_filtering_str != "bad_ts":
            resp = fstore.get_offline_features(
                feature_vector=vec,
                start_time=test_base_time - pd.Timedelta(minutes=3),
                end_time=test_base_time,
                timestamp_for_filtering=timestamp_for_filtering,
                engine="spark",
                run_config=fstore.RunConfig(local=self.run_local, kind="remote-spark"),
                spark_service=self.spark_service,
                target=target,
            )
            res_df = resp.to_dataframe().sort_index(axis=1)

            if not timestamp_for_filtering_str:
                assert res_df["val"].tolist() == [1, 2]
            elif timestamp_for_filtering_str == "other_ts":
                assert res_df["val"].tolist() == [3, 4]

            assert res_df.columns == ["val"]
        else:
            err = (
                mlrun.errors.MLRunInvalidArgumentError
                if self.run_local
                else mlrun.runtimes.utils.RunError
            )
            with pytest.raises(
                err,
                match="Feature set `fs1` does not have a column named `bad_ts` to filter on.",
            ):
                fstore.get_offline_features(
                    feature_vector=vec,
                    start_time=test_base_time - pd.Timedelta(minutes=3),
                    end_time=test_base_time,
                    timestamp_for_filtering=timestamp_for_filtering,
                    engine="spark",
                    run_config=fstore.RunConfig(
                        local=self.run_local, kind="remote-spark"
                    ),
                    spark_service=self.spark_service,
                    target=target,
                )
