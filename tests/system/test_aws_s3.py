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
import random

import pandas as pd
import pytest

import mlrun
import mlrun.errors
import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.targets import ParquetTarget
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
@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestAwsS3(TestMLRunSystem):
    def _make_target_names(
        self, prefix, bucket_name, object_dir, object_file, csv_file
    ):
        bucket_path = prefix + bucket_name
        object_path = f"{object_dir}/{object_file}"
        df_path = f"{object_dir}/{csv_file}"
        object_url = f"{bucket_path}/{object_path}"
        res = {
            "bucket_path": bucket_path,
            "object_path": object_path,
            "df_path": df_path,
            "object_url": object_url,
            "df_url": f"{bucket_path}/{df_path}",
            "blob_url": f"{object_url}.blob",
            "parquet_url": f"{object_url}.parquet",
        }
        return res

    def setup_method(self, method):
        super().setup_method(method)
        self._bucket_name = test_environment["AWS_BUCKET_NAME"]
        self._access_key_id = test_environment["AWS_ACCESS_KEY_ID"]
        self._secret_access_key = test_environment["AWS_SECRET_ACCESS_KEY"]

        object_dir = "test_mlrun_s3_objects"
        object_file = f"file_{random.randint(0, 1000)}.txt"
        csv_file = f"file_{random.randint(0,1000)}.csv"

        self.s3 = {
            "s3": self._make_target_names(
                "s3://", self._bucket_name, object_dir, object_file, csv_file
            ),
            "ds": self._make_target_names(
                "ds://s3ds_profile/",
                self._bucket_name,
                object_dir,
                object_file,
                csv_file,
            ),
        }

        mlrun.get_or_create_project(self.project_name)
        profile = DatastoreProfileS3(
            name="s3ds_profile",
            access_key=self._access_key_id,
            secret_key=self._secret_access_key,
        )
        register_temporary_client_datastore_profile(profile)

    def test_ingest_single_parquet_file(self, use_datastore_profile):
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]

        df1 = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})

        logger.info(f'Object URL: {param["parquet_url"]}')

        targets = [ParquetTarget(path=param["parquet_url"])]

        fset = fstore.FeatureSet(
            name="overwrite-pq-spec-path", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df1, targets=targets)
