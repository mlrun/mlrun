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
from pathlib import Path

import pandas as pd
import pytest
import yaml

import mlrun
import mlrun.errors
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-dbfs-store.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
test_parquet = here / "test_data.parquet"
test_csv = here / "test_data.csv"
with open(test_filename, "r") as f:
    test_string = f.read()

MUST_HAVE_VARIABLES = ["DATABRICKS_TOKEN", "DATABRICKS_WORKSPACE"]


def is_dbfs_configured():
    env_params = config["env"]
    for necessary_variable in MUST_HAVE_VARIABLES:
        if env_params.get(necessary_variable, None) is None:
            return False
    return True


@pytest.mark.skipif(
    not is_dbfs_configured(),
    reason="DBFS storage parameters not configured",
)
class TestDBFSStore:
    def setup_class(self):
        self._databricks_workspace = config["env"].get("DATABRICKS_WORKSPACE")
        self._object_dir = "/test_mlrun_dbfs_objects"
        self._object_file = f"file_{str(uuid.uuid4())}.txt"
        self._object_path = f"{self._object_dir}/{self._object_file}"
        self._dbfs_url = "dbfs://" + self._databricks_workspace
        self._object_url = self._dbfs_url + self._object_path
        self.secrets = {}
        token = config["env"].get("DATABRICKS_TOKEN", None)
        self.secrets["DATABRICKS_TOKEN"] = token
        logger.info(f"Object URL: {self._object_url}")

    def teardown_class(self):
        dir_dataitem = mlrun.run.get_dataitem(
            self._dbfs_url + self._object_dir, secrets=self.secrets
        )
        test_files = dir_dataitem.listdir()
        store = dir_dataitem.store
        for test_file in test_files:
            store.rm(path=f"{self._object_dir}/{test_file}")

    def _perform_dbfs_tests(self, secrets):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(test_string)
        response = data_item.get()
        assert response.decode() == test_string
        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:]
        stat = data_item.stat()
        assert stat.size == len(test_string)

        dir_dataitem = mlrun.run.get_dataitem(self._dbfs_url + self._object_dir)
        dir_list = dir_dataitem.listdir()
        assert self._object_file in dir_list

        source_parquet = pd.read_parquet(test_parquet)
        upload_parquet_file_path = (
            f"{self._object_dir}/file_{str(uuid.uuid4())}.parquet"
        )
        upload_parquet_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_parquet_file_path
        )
        upload_parquet_data_item.upload(str(test_parquet))
        response = upload_parquet_data_item.as_df()
        assert source_parquet.equals(response)
        upload_parquet_data_item.delete()
        with pytest.raises(FileNotFoundError) as file_not_found_error:
            upload_parquet_data_item.stat()
        assert (
            str(file_not_found_error.value)
            == f"No file or directory exists on path {upload_parquet_file_path}."
        )

        source_csv = pd.read_csv(test_csv)
        upload_csv_file_path = f"{self._object_dir}/file_{str(uuid.uuid4())}.csv"
        upload_csv_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_csv_file_path
        )
        upload_csv_data_item.upload(str(test_csv))
        response = upload_csv_data_item.as_df()
        assert source_csv.equals(response)

    def test_secrets_as_input(self):
        self._perform_dbfs_tests(secrets=self.secrets)

    def test_using_dbfs_env_variable(self):
        env_params = config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        self._perform_dbfs_tests(secrets={})
