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

import tempfile
import time
import uuid
from datetime import datetime, timedelta

import fsspec
import pandas as pd
import pytest
import pytz
from pandas.testing import assert_frame_equal

import mlrun
import mlrun.errors
import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import ParquetTarget, get_default_prefix_for_target
from mlrun.utils import logger
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not test_environment.get("AWS_ACCESS_KEY_ID"),
    reason="AWS_ACCESS_KEY_ID is not set",
)
@pytest.mark.skipif(
    not test_environment.get("AWS_SECRET_ACCESS_KEY"),
    reason="AWS_SECRET_ACCESS_KEY is not set",
)
@pytest.mark.skipif(
    not test_environment.get("AWS_BUCKET_NAME"),
    reason="AWS_BUCKET_NAME is not set",
)
class TestAwsS3(TestMLRunSystem):
    project_name = "s3-system-test"
    object_dir = "test_aws_s3"

    def _make_target_names(self, prefix, bucket_name, object_sub_dir):
        bucket_path = prefix + bucket_name
        object_sub_dir_path = f"{self.object_dir}/{object_sub_dir}"
        object_sub_dir_url = f"{bucket_path}/{object_sub_dir_path}"

        parquet_file = f"file_{uuid.uuid4()}.parquet"
        res = {
            "bucket_path": bucket_path,
            "object_sub_dir_path": object_sub_dir_path,
            "object_sub_dir_url": object_sub_dir_url,
            "parquet_url": f"{object_sub_dir_url}/{parquet_file}",
            "parquets_url": f"{object_sub_dir_url}/parquets",
            "test_dir_path": f"{bucket_name}/{self.object_dir}",
        }
        return res

    def setup_method(self, method):
        super().setup_method(method)
        self._bucket_name = test_environment["AWS_BUCKET_NAME"]
        self._access_key_id = test_environment["AWS_ACCESS_KEY_ID"]
        self._secret_access_key = test_environment["AWS_SECRET_ACCESS_KEY"]
        self._endpoint_url = test_environment.get("AWS_ENDPOINT_URL")

        object_sub_dir = f"dir_{uuid.uuid4()}"

        self.s3 = {
            "s3": self._make_target_names("s3://", self._bucket_name, object_sub_dir),
            "ds_with_bucket": self._make_target_names(
                "ds://s3ds_profile_with_bucket",
                "",  # no bucket, since it is part of the ds profile
                object_sub_dir,
            ),
        }

        mlrun.get_or_create_project(self.project_name)
        profile = DatastoreProfileS3(
            name="s3ds_profile_with_bucket",
            access_key_id=self._access_key_id,
            secret_key=self._secret_access_key,
            bucket=self._bucket_name,
            endpoint_url=self._endpoint_url,
        )
        register_temporary_client_datastore_profile(profile)

    def custom_teardown(self):
        s3_fs = fsspec.filesystem(
            "s3",
            key=self._access_key_id,
            secret=self._secret_access_key,
        )
        full_path = self.s3["s3"]["test_dir_path"]
        if s3_fs.exists(full_path):
            files = s3_fs.ls(full_path)
            for file in files:
                s3_fs.rm(file, recursive=True)
            s3_fs.rm(full_path)

    @pytest.mark.parametrize("url_type", ["s3", "ds_with_bucket"])
    @pytest.mark.parametrize("target_path", ["parquets_url", "parquet_url"])
    def test_ingest_with_parquet_source(self, url_type, target_path):
        #  create source
        s3_fs = fsspec.filesystem(
            "s3",
            key=self._access_key_id,
            secret=self._secret_access_key,
            endpoint_url=self._endpoint_url,
        )
        param = self.s3[url_type]
        logger.info(f"Using URL {param['object_sub_dir_url']}")
        data = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
        df = pd.DataFrame(data)
        source_file = f"source_{uuid.uuid4()}.parquet"
        source_url = f"{param['object_sub_dir_url']}/{source_file}"

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            df.to_parquet(temp_file.name)
            path_only = f"{self.s3['s3']['object_sub_dir_path']}/{source_file}"
            s3_fs.put_file(temp_file.name, f"{self._bucket_name}/{path_only}")
        parquet_source = ParquetSource(name="test", path=source_url)

        # ingest
        target = ParquetTarget(path=param[target_path])
        fset = fstore.FeatureSet(
            name="test_fs",
            entities=[fstore.Entity("Column1")],
        )
        fset.set_targets(
            targets=[target],
            with_defaults=False,
        )
        fset.ingest(source=parquet_source)
        target_path = fset.get_target_path()
        result = ParquetSource(path=target_path).to_dataframe(
            columns=("Column1", "Column2")
        )
        result.reset_index(inplace=True, drop=False)

        assert_frame_equal(
            df.sort_index(axis=1), result.sort_index(axis=1), check_like=True
        )

        s3_path = (
            f"{self._bucket_name}/{target_path[target_path.index(self.object_dir) :]}"
        )
        # Check for ML-6587 regression
        assert s3_fs.exists(s3_path)
        fset.purge_targets()
        assert not s3_fs.exists(s3_path)

    def test_ingest_ds_default_target(self):
        s3_fs = fsspec.filesystem(
            "s3",
            key=self._access_key_id,
            secret=self._secret_access_key,
            endpoint_url=self._endpoint_url,
        )
        param = self.s3["ds_with_bucket"]
        logger.info(f"Using URL {param['parquets_url']}")
        data = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
        df = pd.DataFrame(data)
        source_file = f"source_{uuid.uuid4()}.parquet"
        source_url = f"{param['object_sub_dir_url']}/{source_file}"
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            df.to_parquet(temp_file.name)
            path_only = f"{self.s3['s3']['object_sub_dir_path']}/{source_file}"
            s3_fs.put_file(temp_file.name, f"{self._bucket_name}/{path_only}")

        parquet_source = ParquetSource(name="test", path=source_url)

        target = ParquetTarget(path="ds://s3ds_profile_with_bucket")
        fset = fstore.FeatureSet(
            name="test_fs",
            entities=[fstore.Entity("Column1")],
        )

        fset.ingest(source=parquet_source, targets=[target])

        expected_default_ds_data_prefix = get_default_prefix_for_target(
            "dsnosql"
        ).format(
            ds_profile_name="s3ds_profile_with_bucket",
            project=fset.metadata.project,
            kind=target.kind,
            name=fset.metadata.name,
        )

        assert fset.get_target_path().startswith(expected_default_ds_data_prefix)

        result = ParquetSource(path=fset.get_target_path()).to_dataframe(
            columns=("Column1", "Column2")
        )
        result.reset_index(inplace=True, drop=False)

        assert_frame_equal(
            df.sort_index(axis=1), result.sort_index(axis=1), check_like=True
        )

        # check our user protection against direct target.purge call in the
        # case of default target + ds (it could delete the whole bucket).
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Unable to delete target. Please Use purge_targets from FeatureSet object.",
        ):
            target.purge()
        assert s3_fs.ls(f"{self._bucket_name}")

    @pytest.mark.parametrize(
        ("partition_keys", "granularity"),
        [
            (["year"], "year"),
            (["year", "month"], "month"),
            (["year", "month", "day"], "day"),
            (["year", "month", "day", "hour"], "hour"),
        ],
    )
    @pytest.mark.parametrize("with_tz", [True, False])
    # Copied from test_feature_store.py for equivalent testing on S3
    # ML-11732
    def test_partitioned_parquet_as_df_time_filtering_optimization(
        self, partition_keys, granularity, with_tz
    ):
        """
        test reading partitioned parquet target as_df method with time filtering
        covers:
          - Partitioned parquet writing via ParquetTarget
          - Reading & filtering via start_time/end_time
          - Empty/out-of-range case
        """
        key = "patient_id"
        base_time = datetime(2020, 12, 1, 17, 0)
        if with_tz:
            base_time = base_time.replace(tzinfo=pytz.UTC)

        df = pd.DataFrame(
            [
                {
                    key: i + 1,
                    "timestamp": base_time + timedelta(hours=i),
                    "value": i * 10,
                }
                for i in range(4)
            ]
        )

        run_id = uuid.uuid4()
        target_path = f"s3://{self._bucket_name}/partition_test_{run_id}"

        target = ParquetTarget(
            name="parquet_target",
            path=target_path,
            partitioned=True,
            time_partitioning_granularity=granularity,
        )

        start_time = base_time + timedelta(hours=1)
        end_time = base_time + timedelta(hours=2, minutes=1)

        target.write_dataframe(df, timestamp_key="timestamp")

        expected_df = df[
            (df["timestamp"] > start_time) & (df["timestamp"] <= end_time)
        ].copy()

        result_df = target.as_df(
            start_time=start_time,
            end_time=end_time,
            time_column="timestamp",
        )

        if with_tz:
            result_df["timestamp"] = (
                pd.to_datetime(result_df["timestamp"])
                .dt.tz_convert("UTC")
                .astype("datetime64[ns, UTC]")
            )
        else:
            result_df["timestamp"] = pd.to_datetime(result_df["timestamp"]).astype(
                "datetime64[ns]"
            )

        result_df = result_df.sort_values(key).reset_index(drop=True)
        expected_df = expected_df.sort_values(key).reset_index(drop=True)
        assert_frame_equal(result_df, expected_df)

        large_base_period_start = base_time - timedelta(days=365)
        large_base_period_end = base_time + timedelta(days=1)
        start = time.monotonic()
        result_df = target.as_df(
            start_time=large_base_period_start,
            end_time=large_base_period_end,
            time_column="timestamp",
        )
        end = time.monotonic()
        assert end - start < 10, "Reading large period took too long"
        if with_tz:
            result_df["timestamp"] = (
                pd.to_datetime(result_df["timestamp"])
                .dt.tz_convert("UTC")
                .astype("datetime64[ns, UTC]")
            )
        else:
            result_df["timestamp"] = pd.to_datetime(result_df["timestamp"]).astype(
                "datetime64[ns]"
            )

        result_df = result_df.sort_values(key).reset_index(drop=True)
        assert_frame_equal(result_df, df.sort_values(key).reset_index(drop=True))

        late_start = base_time + timedelta(days=2)
        late_end = late_start + timedelta(days=1)

        empty_df = target.as_df(
            start_time=late_start,
            end_time=late_end,
            time_column="timestamp",
        )
        assert empty_df.empty, "df should be empty for out-of-range time filter"
