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
import uuid

import fsspec
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileGCS,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.targets import ParquetTarget
from mlrun.utils import logger
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not test_environment.get("GOOGLE_APPLICATION_CREDENTIALS"),
    reason="GOOGLE_APPLICATION_CREDENTIALS is not set",
)
@pytest.mark.skipif(
    not test_environment.get("GCS_BUCKET_NAME"),
    reason="GCS_BUCKET_NAME is not set",
)
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestGoogleCloudStorage(TestMLRunSystem):
    @classmethod
    def clean_test_directory(cls):
        test_dir = f"{cls._bucket_name}/{cls.test_dir}"
        if cls._gcs_fs.exists(test_dir):
            cls._gcs_fs.delete(test_dir, recursive=True)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls._bucket_name = test_environment["GCS_BUCKET_NAME"]
        cls.credentials_path = test_environment["GOOGLE_APPLICATION_CREDENTIALS"]
        cls.test_dir = "test_mlrun_gcs_system_objects"
        cls.profile_name = "gcs_system_profile"
        cls._gcs_fs = fsspec.filesystem("gcs", token=cls.credentials_path)
        cls.clean_test_directory()

    @classmethod
    def teardown_class(cls):
        super().teardown_class()
        cls.clean_test_directory()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self._object_dir = self.test_dir + "/" + f"target_directory_{uuid.uuid4()}"
        self._bucket_path = (
            f"ds://{self.profile_name}/{self._bucket_name}"
            if use_datastore_profile
            else "gcs://" + self._bucket_name
        )
        self._target_url = self._bucket_path + "/" + self._object_dir
        logger.info(f"Object URL: {self._target_url}")
        if use_datastore_profile:
            kwargs = {"credentials_path": self.credentials_path}
            profile = DatastoreProfileGCS(name=self.profile_name, **kwargs)
            register_temporary_client_datastore_profile(profile)
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

    def test_ingest_single_parquet_file(self, use_datastore_profile):
        df = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        targets = [ParquetTarget(path=self._target_url)]

        fset = fstore.FeatureSet(
            name="gcs_system_test", entities=[fstore.Entity("name")]
        )
        fstore.ingest(fset, df, targets=targets)
        result = targets[0].as_df()
        result.reset_index(inplace=True, drop=False)

        assert_frame_equal(
            df.sort_index(axis=1), result.sort_index(axis=1), check_like=True
        )
