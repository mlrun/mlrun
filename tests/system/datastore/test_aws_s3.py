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
import uuid

import fsspec
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import mlrun
import mlrun.errors
import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import ParquetTarget
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
    project_name = "s3-system-test"

    def _make_target_names(self, prefix, bucket_name, object_dir, object_file):
        bucket_path = prefix + bucket_name
        object_path = f"{object_dir}/{object_file}"
        object_url = f"{bucket_path}/{object_path}"
        res = {
            "bucket_path": bucket_path,
            "object_path": object_path,
            "object_url": object_url,
            "parquet_url": f"{object_url}.parquet",
            "test_dir": f"{bucket_path}/{object_dir}",
        }
        return res

    def setup_method(self, method):
        super().setup_method(method)
        self._bucket_name = test_environment["AWS_BUCKET_NAME"]
        self._access_key_id = test_environment["AWS_ACCESS_KEY_ID"]
        self._secret_access_key = test_environment["AWS_SECRET_ACCESS_KEY"]

        object_dir = "test_aws_s3"
        object_file = f"file_{uuid.uuid4()}"

        self.s3 = {
            "s3": self._make_target_names(
                "s3://", self._bucket_name, object_dir, object_file
            ),
            "ds": self._make_target_names(
                "ds://s3ds_profile/",
                self._bucket_name,
                object_dir,
                object_file,
            ),
        }

        mlrun.get_or_create_project(self.project_name)
        profile = DatastoreProfileS3(
            name="s3ds_profile",
            access_key_id=self._access_key_id,
            secret_key=self._secret_access_key,
        )
        register_temporary_client_datastore_profile(profile)

    def custom_teardown(self):
        s3_fs = fsspec.filesystem(
            "s3", key=self._access_key_id, secret=self._secret_access_key
        )
        full_path = self.s3["s3"]["test_dir"]
        if s3_fs.exists(full_path):
            files = s3_fs.ls(full_path)
            for file in files:
                s3_fs.rm(file)
            s3_fs.rm(full_path)

    def test_ingest_with_parquet_source(self, use_datastore_profile):
        #  create source
        s3_fs = fsspec.filesystem(
            "s3", key=self._access_key_id, secret=self._secret_access_key
        )
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        print(f"Using URL {param['parquet_url']}\n")
        data = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
        df = pd.DataFrame(data)
        source_path = param["parquet_url"]
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            df.to_parquet(temp_file.name)
            path_only = source_path.replace("ds://s3ds_profile/", "").replace(
                "s3://", ""
            )
            s3_fs.put_file(temp_file.name, path_only)
        parquet_source = ParquetSource(name="test", path=source_path)

        # ingest
        target_path = f"{os.path.dirname(param['parquet_url'])}/target_{uuid.uuid4()}"
        targets = [ParquetTarget(path=target_path)]
        fset = fstore.FeatureSet(
            name="test_fs",
            entities=[fstore.Entity("Column1")],
        )

        fset.ingest(source=parquet_source, targets=targets)
        result = ParquetSource(path=target_path).to_dataframe(
            columns=("Column1", "Column2")
        )
        result.reset_index(inplace=True, drop=False)

        assert_frame_equal(
            df.sort_index(axis=1), result.sort_index(axis=1), check_like=True
        )
