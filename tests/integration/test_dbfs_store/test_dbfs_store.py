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
from typing import List

import pandas as pd
import pytest
import yaml
from databricks.sdk import WorkspaceClient

import mlrun
import mlrun.errors
from mlrun.utils import logger

PARQUETS_DIR = "/parquets"
CSV_DIR = "/csv"
here = Path(__file__).absolute().parent
config_file_path = here / "test-dbfs-store.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_file_path = here / "test.txt"
parquet_path = here / "test_data.parquet"
additional_parquet_path = here / "additional_data.parquet"
csv_path = here / "test_data.csv"
additional_csv_path = here / "additional_data.csv"
with open(test_file_path, "r") as f:
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
        self.test_root_dir = "/test_mlrun_dbfs_objects"
        self._object_file = f"file_{str(uuid.uuid4())}.txt"
        self._object_path = f"{self.test_root_dir}/{self._object_file}"
        self._dbfs_url = "dbfs://" + self._databricks_workspace
        self._object_url = self._dbfs_url + self._object_path
        self.secrets = {}
        token = config["env"].get("DATABRICKS_TOKEN", None)
        self.secrets["DATABRICKS_TOKEN"] = token
        self.parquets_dir = PARQUETS_DIR
        self.csv_dir = CSV_DIR
        self.workspace = WorkspaceClient(host=self._databricks_workspace, token=token)
        logger.info(f"Object URL: {self._object_url}")

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        all_paths = [file_info.path for file_info in self.workspace.dbfs.list("/")]
        if self.test_root_dir not in all_paths:
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{self.parquets_dir}")
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{self.csv_dir}")
        else:
            self.workspace.dbfs.delete(f"{self.test_root_dir}", recursive=True)
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{self.parquets_dir}")
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{self.csv_dir}")

    def teardown_class(self):
        dir_dataitem = mlrun.run.get_dataitem(
            self._dbfs_url + self.test_root_dir, secrets=self.secrets
        )
        test_files = dir_dataitem.listdir()
        store = dir_dataitem.store
        for test_file in test_files:
            store.rm(path=f"{self.test_root_dir}/{test_file}", recursive=True)

    def _perform_dbfs_tests(self, secrets):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(test_string)
        response = data_item.get()
        assert response.decode() == test_string
        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:]
        stat = data_item.stat()
        assert stat.size == len(test_string)

        dir_dataitem = mlrun.run.get_dataitem(
            self._dbfs_url + self.test_root_dir, secrets=secrets
        )
        dir_list = dir_dataitem.listdir()
        assert self._object_file in dir_list

        source_parquet = pd.read_parquet(parquet_path)
        upload_parquet_file_path = (
            f"{self.test_root_dir}/file_{str(uuid.uuid4())}.parquet"
        )
        upload_parquet_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_parquet_file_path, secrets=secrets
        )
        upload_parquet_data_item.upload(str(parquet_path))
        response = upload_parquet_data_item.as_df()
        assert source_parquet.equals(response)
        upload_parquet_data_item.delete()
        with pytest.raises(FileNotFoundError) as file_not_found_error:
            upload_parquet_data_item.stat()
        assert (
            str(file_not_found_error.value)
            == f"No file or directory exists on path {upload_parquet_file_path}."
        )

        source_csv = pd.read_csv(csv_path)
        upload_csv_file_path = f"{self.test_root_dir}/file_{str(uuid.uuid4())}.csv"
        upload_csv_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_csv_file_path
        )
        upload_csv_data_item.upload(str(csv_path))
        response = upload_csv_data_item.as_df()
        assert source_csv.equals(response)

        source_json = pd.read_json(json_path)
        upload_json_file_path = f"{self.test_root_dir}/file_{str(uuid.uuid4())}.json"
        upload_json_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_json_file_path
        )
        upload_json_data_item.upload(str(json_path))
        response = upload_json_data_item.as_df()
        assert source_json.equals(response)

    @pytest.mark.parametrize(
        "directory, file_format, file_extension, files_paths, reader",
        [
            (
                PARQUETS_DIR,
                "parquet",
                "parquet",
                [parquet_path, additional_parquet_path],
                pd.read_parquet,
            ),
            (CSV_DIR, "csv", "csv", [csv_path, additional_csv_path], pd.read_csv),
        ],
    )
    def test_check_read_df_dir(
        self,
        directory: str,
        file_format: str,
        file_extension: str,
        files_paths: List[Path],
        reader: callable,
    ):
        first_file_path = str(files_paths[0])
        second_file_path = str(files_paths[1])
        uploaded_file_path = (
            f"{self.test_root_dir}{directory}/file_{str(uuid.uuid4())}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + uploaded_file_path, secrets=self.secrets
        )
        uploaded_data_item.upload(first_file_path)

        uploaded_file_path = (
            f"{self.test_root_dir}{directory}/file_{str(uuid.uuid4())}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + uploaded_file_path, secrets=self.secrets
        )
        uploaded_data_item.upload(second_file_path)

        dir_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + os.path.dirname(uploaded_file_path), secrets=self.secrets
        )
        response_df = (
            dir_data_item.as_df(format=file_format)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        df = reader(first_file_path)
        additional_df = reader(second_file_path)
        appended_df = (
            pd.concat([df, additional_df], axis=0)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        assert response_df.equals(appended_df)

    def test_secrets_as_input(self):
        self._perform_dbfs_tests(secrets=self.secrets)

    def test_using_dbfs_env_variable(self):
        env_params = config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        self._perform_dbfs_tests(secrets={})
