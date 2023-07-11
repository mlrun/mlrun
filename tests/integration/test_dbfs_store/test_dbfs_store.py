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

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from databricks.sdk import WorkspaceClient

import mlrun
import mlrun.errors

PARQUETS_DIR = "/parquets"
CSV_DIR = "/csv"
here = Path(__file__).absolute().parent
config_file_path = here / "test-dbfs-store.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_file_path = str(here / "test.txt")
json_path = str(here / "test_data.json")
parquet_path = str(here / "test_data.parquet")
additional_parquet_path = str(here / "additional_data.parquet")
csv_path = str(here / "test_data.csv")
additional_csv_path = str(here / "additional_data.csv")
with open(test_file_path, "r") as f:
    test_string = f.read()

MUST_HAVE_VARIABLES = ["DATABRICKS_TOKEN", "DATABRICKS_HOST"]


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
        databricks_host = config["env"].get("DATABRICKS_HOST")
        env_params = config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        self.test_root_dir = "/test_mlrun_dbfs_objects"
        self._dbfs_url = "dbfs://" + databricks_host
        self.workspace = WorkspaceClient()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        all_paths = [file_info.path for file_info in self.workspace.dbfs.list("/")]
        if self.test_root_dir not in all_paths:
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{PARQUETS_DIR}")
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{CSV_DIR}")
        else:
            self.workspace.dbfs.delete(self.test_root_dir, recursive=True)
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{PARQUETS_DIR}")
            self.workspace.dbfs.mkdirs(f"{self.test_root_dir}{CSV_DIR}")

    def teardown_class(self):
        all_paths_under_test_root = [
            file_info.path for file_info in self.workspace.dbfs.list(self.test_root_dir)
        ]
        for path in all_paths_under_test_root:
            self.workspace.dbfs.delete(path, recursive=True)

    def _get_data_item(self, secrets={}):
        object_path = f"{self.test_root_dir}/file_{uuid.uuid4()}.txt"
        object_url = f"{self._dbfs_url}{object_path}"
        return mlrun.run.get_dataitem(object_url, secrets=secrets), object_url

    @pytest.mark.parametrize("use_secrets_as_parameters", [True, False])
    def test_put_and_get(self, use_secrets_as_parameters):
        secrets = {}
        if use_secrets_as_parameters:
            token = config["env"].get("DATABRICKS_TOKEN", None)
            secrets = {"DATABRICKS_TOKEN": token}
            os.environ["DATABRICKS_TOKEN"] = ""
        try:
            data_item, _ = self._get_data_item(secrets=secrets)
            data_item.put(test_string)
            response = data_item.get()
            assert response.decode() == test_string

            response = data_item.get(offset=20)
            assert response.decode() == test_string[20:]

        finally:
            if use_secrets_as_parameters:
                os.environ["DATABRICKS_TOKEN"] = token

    def test_stat(self):
        data_item, _ = self._get_data_item()
        data_item.put(test_string)
        stat = data_item.stat()
        assert stat.size == len(test_string)

    def test_list_dir(self):
        data_item, object_url = self._get_data_item()
        data_item.put(test_string)
        dir_dataitem = mlrun.run.get_dataitem(
            self._dbfs_url + self.test_root_dir,
        )
        dir_list = dir_dataitem.listdir()
        assert object_url.split("/")[-1] in dir_list

    def test_upload(self):
        data_item, _ = self._get_data_item()
        data_item.upload(test_file_path)
        response = data_item.get()
        assert response.decode() == test_string

    def test_rm(self):
        data_item, _ = self._get_data_item()
        data_item.upload(test_file_path)
        data_item.stat()
        data_item.delete()
        with pytest.raises(FileNotFoundError) as file_not_found_error:
            data_item.stat()
        assert "No file or directory exists on path" in str(file_not_found_error.value)

    @pytest.mark.parametrize(
        "file_extension, local_file_path, reader",
        [
            (
                "parquet",
                parquet_path,
                pd.read_parquet,
            ),
            ("csv", csv_path, pd.read_csv),
            ("json", json_path, pd.read_json),
        ],
    )
    def test_as_df(self, file_extension: str, local_file_path: str, reader: callable):
        source = reader(local_file_path)
        upload_file_path = f"{self.test_root_dir}/file_{uuid.uuid4()}.{file_extension}"
        upload_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_file_path,
        )
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df()
        assert source.equals(response)

    @pytest.mark.parametrize(
        "file_extension, local_file_path, reader",
        [
            (
                "parquet",
                parquet_path,
                dd.read_parquet,
            ),
            ("csv", csv_path, dd.read_csv),
            ("json", json_path, dd.read_json),
        ],
    )
    def test_as_df_dd(
        self, file_extension: str, local_file_path: str, reader: callable
    ):
        source = reader(local_file_path)
        upload_file_path = f"{self.test_root_dir}/file_{uuid.uuid4()}.{file_extension}"
        upload_data_item = mlrun.run.get_dataitem(
            self._dbfs_url + upload_file_path,
        )
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd)
        assert dd.assert_eq(source, response)

    def _setup_df_dir(
        self, first_file_path, second_file_path, file_extension, directory
    ):
        uploaded_file_path = (
            f"{self.test_root_dir}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(self._dbfs_url + uploaded_file_path)
        uploaded_data_item.upload(first_file_path)

        uploaded_file_path = (
            f"{self.test_root_dir}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(self._dbfs_url + uploaded_file_path)
        uploaded_data_item.upload(second_file_path)
        return os.path.dirname(uploaded_file_path)

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
        first_file_path = files_paths[0]
        second_file_path = files_paths[1]
        df_dir = self._setup_df_dir(
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
            directory=directory,
        )
        dir_data_item = mlrun.run.get_dataitem(self._dbfs_url + df_dir)
        response_df = (
            dir_data_item.as_df(format=file_format)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        df = reader(files_paths[0])
        additional_df = reader(second_file_path)
        appended_df = (
            pd.concat([df, additional_df], axis=0)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        assert response_df.equals(appended_df)

    @pytest.mark.parametrize(
        "directory, file_format, file_extension, files_paths, reader",
        [
            (
                PARQUETS_DIR,
                "parquet",
                "parquet",
                [parquet_path, additional_parquet_path],
                dd.read_parquet,
            ),
            (CSV_DIR, "csv", "csv", [csv_path, additional_csv_path], dd.read_csv),
        ],
    )
    def test_check_read_df_dir_dd(
        self,
        directory: str,
        file_format: str,
        file_extension: str,
        files_paths: List[Path],
        reader: callable,
    ):
        first_file_path = files_paths[0]
        second_file_path = files_paths[1]
        df_dir = self._setup_df_dir(
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
            directory=directory,
        )
        dir_data_item = mlrun.run.get_dataitem(self._dbfs_url + df_dir)
        response_df = (
            dir_data_item.as_df(format=file_format, df_module=dd)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        df = reader(first_file_path)
        additional_df = reader(second_file_path)
        appended_df = (
            dd.concat([df, additional_df], axis=0)
            .sort_values("Name")
            .reset_index(drop=True)
        )
        assert dd.assert_eq(appended_df, response_df)
