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
import json
import os
import tempfile
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
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import CSVTarget, ParquetTarget
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
    project_name = "gcsfs-system-test"

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
        try:
            credentials = json.loads(cls.credentials_path)
            token = credentials
        except json.JSONDecodeError:
            token = cls.credentials_path
        cls._gcs_fs = fsspec.filesystem("gcs", token=token, use_listings_cache=False)
        cls.clean_test_directory()

    @classmethod
    def teardown_class(cls):
        super().teardown_class()
        cls.clean_test_directory()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self._object_dir = self.test_dir + "/" + f"target_directory_{uuid.uuid4()}"
        self._bucket_path = (
            f"ds://{self.profile_name}"
            if use_datastore_profile
            else "gcs://" + self._bucket_name
        )
        self._source_url_template = (
            self._bucket_path + "/" + self._object_dir + "/source"
        )
        self._target_url_template = (
            self._bucket_path + "/" + self._object_dir + "/target"
        )
        logger.info(f"Object URL template: {self._target_url_template}")
        if use_datastore_profile:
            kwargs = {"credentials_path": self.credentials_path}
            profile = DatastoreProfileGCS(
                name=self.profile_name, bucket=self._bucket_name, **kwargs
            )
            register_temporary_client_datastore_profile(profile)
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

    @pytest.mark.parametrize(
        "source_class, target_class, file_extension, reader, writer, writer_kwargs, reset_index, use_folder",
        [
            (
                CSVSource,
                CSVTarget,
                ".csv",
                pd.read_csv,
                pd.DataFrame.to_csv,
                {"index": False},
                False,
                False,
            ),
            (
                ParquetSource,
                ParquetTarget,
                ".parquet",
                pd.read_parquet,
                pd.DataFrame.to_parquet,
                {},
                True,
                False,
            ),
            (
                ParquetSource,
                ParquetTarget,
                ".parquet",
                pd.read_parquet,
                pd.DataFrame.to_parquet,
                {},
                True,
                True,
            ),
        ],
    )
    def test_ingest_single_file(
        self,
        use_datastore_profile,
        source_class,
        target_class,
        file_extension,
        reader,
        writer,
        writer_kwargs,
        reset_index,
        use_folder,
    ):
        df = pd.DataFrame({"name": ["ABC", "DEF", "GHI"], "value": [1, 2, 3]})
        source_url = self._source_url_template + file_extension
        target_url = (
            self._target_url_template
            if use_folder
            else self._target_url_template + file_extension
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"{file_extension}", delete=True
        ) as df_file:
            writer(df, df_file.name, **writer_kwargs)
            self._gcs_fs.upload(
                lpath=df_file.name,
                rpath=source_url.replace(self._bucket_path, self._bucket_name),
            )
        source = source_class(path=source_url)
        targets = [target_class(path=target_url)]

        fset = fstore.FeatureSet(
            name="gcs_system_test", entities=[fstore.Entity("name")]
        )
        fset.set_targets(
            targets=targets,
            with_defaults=False,
        )
        fset.ingest(source)
        target_path = fset.get_target_path()

        # Avoids adding date columns when using a folder as the target.
        to_dataframe_dict = {"columns": list(df.columns)} if use_folder else {}
        result = source_class(path=target_path).to_dataframe(**to_dataframe_dict)
        if reset_index:
            result.reset_index(inplace=True, drop=False)
        assert_frame_equal(
            df.sort_index(axis=1), result.sort_index(axis=1), check_like=True
        )

        gcs_path = (
            f"{self._bucket_name}/{target_path[target_path.index(self.test_dir):]}"
        )
        # Check for ML-6587 regression
        assert self._gcs_fs.exists(gcs_path)
        fset.purge_targets()
        assert not self._gcs_fs.exists(gcs_path)
