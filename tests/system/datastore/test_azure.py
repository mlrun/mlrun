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

import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileAzureBlob,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import CSVTarget, ParquetTarget
from mlrun.utils import logger
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not test_environment.get("AZURE_STORAGE_CONNECTION_STRING"),
    reason="AZURE_STORAGE_CONNECTION_STRING is not set",
)
@pytest.mark.skipif(
    not test_environment.get("AZURE_CONTAINER"),
    reason="AZURE_CONTAINER is not set",
)
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestAzureBlobSystem(TestMLRunSystem):
    project_name = "azure-blob-system-test"

    @classmethod
    def clean_test_directory(cls):
        test_dir = f"{cls._bucket_name}/{cls.test_dir}"
        if cls._azure_fs.exists(test_dir):
            cls._azure_fs.delete(test_dir, recursive=True)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls._bucket_name = test_environment["AZURE_CONTAINER"]
        cls.connection_string = test_environment["AZURE_STORAGE_CONNECTION_STRING"]
        cls.test_dir = "test_mlrun_azure_system_objects"
        cls.profile_name = "azure_system_profile"
        cls._azure_fs = fsspec.filesystem(
            "az", using_bucket=cls._bucket_name, connection_string=cls.connection_string
        )
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
            else "az://" + self._bucket_name
        )
        self._source_url_template = (
            self._bucket_path + "/" + self._object_dir + "/source"
        )
        self._target_url_template = (
            self._bucket_path + "/" + self._object_dir + "/target"
        )
        logger.info(f"Object URL template: {self._target_url_template}")
        if use_datastore_profile:
            kwargs = {"connection_string": self.connection_string}
            profile = DatastoreProfileAzureBlob(
                name=self.profile_name, container=self._bucket_name, **kwargs
            )
            register_temporary_client_datastore_profile(profile)
            os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
        else:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = self.connection_string

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
            self._azure_fs.upload(
                lpath=df_file.name,
                rpath=source_url.replace(self._bucket_path, self._bucket_name),
            )
        source = source_class(path=source_url)
        targets = [target_class(path=target_url)]
        fset = fstore.FeatureSet(
            name="az_system_test", entities=[fstore.Entity("name")]
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

        azure_path = (
            f"{self._bucket_name}/{target_path[target_path.index(self.test_dir):]}"
        )
        # Check for ML-6587 regression
        assert self._azure_fs.exists(azure_path)
        fset.purge_targets()
        assert not self._azure_fs.exists(azure_path)
