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

import pandas as pd
import pytest
from databricks.sdk import WorkspaceClient
from pandas.testing import assert_frame_equal

import mlrun.feature_store as fstore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileDBFS,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.sources import CSVSource, ParquetSource
from mlrun.datastore.targets import CSVTarget, ParquetTarget
from mlrun.feature_store import Entity
from mlrun.utils import logger
from tests.datastore.databricks_utils import (
    MLRUN_ROOT_DIR,
    setup_dbfs_dirs,
    teardown_dbfs_dirs,
)
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not test_environment.get("DATABRICKS_TOKEN"),
    reason="DATABRICKS_TOKEN is not set",
)
@pytest.mark.skipif(
    not test_environment.get("DATABRICKS_HOST"),
    reason="DATABRICKS_HOST is not set",
)
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestDBFS(TestMLRunSystem):
    project_name = "dbfs-system-test"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.workspace = WorkspaceClient()
        cls.test_dir = "/dbfs_system_test"
        cls.dir_path = f"{MLRUN_ROOT_DIR}{cls.test_dir}"
        cls.profile_name = "dbfs_system_test_profile"
        cls.token = os.environ.pop("DATABRICKS_TOKEN", None)
        cls.host = os.environ.pop("DATABRICKS_HOST", None)

    @classmethod
    def teardown_class(cls):
        super().teardown_class()
        teardown_dbfs_dirs(
            workspace=cls.workspace, specific_test_class_dir=cls.test_dir
        )

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self._object_dir = f"/directory_{uuid.uuid4()}"
        self._object_dir_path = self.dir_path + self._object_dir
        if use_datastore_profile:
            profile = DatastoreProfileDBFS(
                name=self.profile_name, endpoint_url=self.host, token=self.token
            )
            register_temporary_client_datastore_profile(profile)
            os.environ.pop("DATABRICKS_TOKEN", None)
            os.environ.pop("DATABRICKS_HOST", None)
            self._url_prefix = f"ds://{self.profile_name}"
        else:
            os.environ["DATABRICKS_TOKEN"] = self.token
            os.environ["DATABRICKS_HOST"] = self.host
            self._url_prefix = "dbfs://"

        self._dir_url = self._url_prefix + self._object_dir_path
        setup_dbfs_dirs(
            workspace=self.workspace,
            specific_test_class_dir=self.test_dir,
            subdirs=[self._object_dir],
        )
        logger.info(f"Object URL: {self._dir_url}")

    @pytest.mark.parametrize(
        "source_class, target_class, local_filename, reader, reader_kwargs, drop_index",
        [
            (
                CSVSource,
                CSVTarget,
                "test_data.csv",
                pd.read_csv,
                {"parse_dates": ["date_of_birth"]},
                False,
            ),
            (
                ParquetSource,
                ParquetTarget,
                "test_data.parquet",
                pd.read_parquet,
                {},
                True,
            ),
        ],
    )
    def test_ingest_with_dbfs(
        self,
        source_class,
        target_class,
        local_filename,
        reader,
        reader_kwargs,
        drop_index,
        use_datastore_profile,
    ):
        local_source_path = os.path.relpath(str(self.assets_path / local_filename))
        key = "name"

        expected = reader(local_source_path, **reader_kwargs)

        measurements = fstore.FeatureSet("measurements", entities=[Entity(key)])

        dbfs_source_path = f"{self._dir_url}/source_{local_filename}"
        dbfs_target_path = f"{self._dir_url}/target_{local_filename}"
        target = target_class(name="specified-path", path=dbfs_target_path)

        with open(local_source_path, "rb") as source_file:
            source_content = source_file.read()
        with self.workspace.dbfs.open(
            f"{self._object_dir_path}/source_{local_filename}",
            write=True,
            overwrite=True,
        ) as f:
            f.write(source_content)
        source = source_class("my_source", dbfs_source_path, **reader_kwargs)
        measurements.ingest(source=source, targets=[target])
        target_file_path = measurements.get_target_path()
        result = source_class(path=target_file_path, **reader_kwargs).to_dataframe()
        if drop_index:
            result.reset_index(inplace=True, drop=False)

        assert_frame_equal(
            expected.sort_index(axis=1),
            result.sort_index(axis=1),
            check_like=True,
            check_dtype=False,
        )
