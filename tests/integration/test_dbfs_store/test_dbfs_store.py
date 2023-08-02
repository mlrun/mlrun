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
from tests.datastore.databricks_utils import (
    MLRUN_ROOT_DIR,
    is_databricks_configured,
    setup_dbfs_dirs,
    teardown_dbfs_dirs,
)


@pytest.mark.skipif(
    not is_databricks_configured(
        Path(__file__).absolute().parent / "test-dbfs-store.yml"
    ),
    reason="DBFS storage parameters not configured",
)
class TestDBFSStore:
    here = Path(__file__).absolute().parent
    config_file_path = here / "test-dbfs-store.yml"
    with config_file_path.open() as fp:
        config = yaml.safe_load(fp)
    parquets_dir = "/parquets"
    csv_dir = "/csv"
    test_file_path = str(here / "test.txt")
    json_path = str(here / "test_data.json")
    parquet_path = str(here / "test_data.parquet")
    additional_parquet_path = str(here / "additional_data.parquet")
    csv_path = str(here / "test_data.csv")
    additional_csv_path = str(here / "additional_data.csv")

    def setup_class(self):
        env_params = self.config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        self.dbfs_store_dir = "/dbfs_store"
        self.dbfs_store_path = f"{MLRUN_ROOT_DIR}{self.dbfs_store_dir}"
        self._dbfs_schema = "dbfs://"
        self.workspace = WorkspaceClient()

        with open(self.test_file_path, "r") as f:
            self.test_string = f.read()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        setup_dbfs_dirs(
            workspace=self.workspace,
            specific_test_class_dir=self.dbfs_store_dir,
            subdirs=[self.parquets_dir, self.csv_dir],
        )

    def teardown_class(self):
        teardown_dbfs_dirs(
            workspace=self.workspace, specific_test_class_dir=self.dbfs_store_dir
        )

    def _get_data_item(self, secrets={}):
        object_path = f"{self.dbfs_store_path}/file_{uuid.uuid4()}.txt"
        object_url = f"{self._dbfs_schema}{object_path}"
        return mlrun.run.get_dataitem(object_url, secrets=secrets), object_url

    @pytest.mark.parametrize("use_secrets_as_parameters", [True, False])
    def test_put_and_get(self, use_secrets_as_parameters):
        secrets = {}
        if use_secrets_as_parameters:
            token = self.config["env"].get("DATABRICKS_TOKEN", None)
            secrets = {"DATABRICKS_TOKEN": token}
            os.environ["DATABRICKS_TOKEN"] = ""
        try:
            data_item, _ = self._get_data_item(secrets=secrets)
            data_item.put(self.test_string)
            response = data_item.get()
            assert response.decode() == self.test_string

            response = data_item.get(offset=20)
            assert response.decode() == self.test_string[20:]

        finally:
            if use_secrets_as_parameters:
                os.environ["DATABRICKS_TOKEN"] = token

    def test_stat(self):
        data_item, _ = self._get_data_item()
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    def test_list_dir(self):
        data_item, object_url = self._get_data_item()
        data_item.put(self.test_string)
        dir_dataitem = mlrun.run.get_dataitem(
            self._dbfs_schema + self.dbfs_store_path,
        )
        dir_list = dir_dataitem.listdir()
        assert object_url.split("/")[-1] in dir_list

    def test_upload(self):
        data_item, _ = self._get_data_item()
        data_item.upload(self.test_file_path)
        response = data_item.get()
        assert response.decode() == self.test_string

    def test_rm(self):
        data_item, _ = self._get_data_item()
        data_item.upload(self.test_file_path)
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
        upload_file_path = (
            f"{self.dbfs_store_path}/file_{uuid.uuid4()}.{file_extension}"
        )
        upload_data_item = mlrun.run.get_dataitem(
            self._dbfs_schema + upload_file_path,
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
        upload_file_path = (
            f"{self.dbfs_store_path}/file_{uuid.uuid4()}.{file_extension}"
        )
        upload_data_item = mlrun.run.get_dataitem(
            self._dbfs_schema + upload_file_path,
        )
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd)
        assert dd.assert_eq(source, response)

    def _setup_df_dir(
        self, first_file_path, second_file_path, file_extension, directory
    ):
        uploaded_file_path = (
            f"{self.dbfs_store_path}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(
            self._dbfs_schema + uploaded_file_path
        )
        uploaded_data_item.upload(first_file_path)

        uploaded_file_path = (
            f"{self.dbfs_store_path}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        uploaded_data_item = mlrun.run.get_dataitem(
            self._dbfs_schema + uploaded_file_path
        )
        uploaded_data_item.upload(second_file_path)
        return os.path.dirname(uploaded_file_path)

    @pytest.mark.parametrize(
        "directory, file_format, file_extension, files_paths, reader",
        [
            (
                parquets_dir,
                "parquet",
                "parquet",
                [parquet_path, additional_parquet_path],
                pd.read_parquet,
            ),
            (csv_dir, "csv", "csv", [csv_path, additional_csv_path], pd.read_csv),
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
        dir_data_item = mlrun.run.get_dataitem(self._dbfs_schema + df_dir)
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
                parquets_dir,
                "parquet",
                "parquet",
                [parquet_path, additional_parquet_path],
                dd.read_parquet,
            ),
            (csv_dir, "csv", "csv", [csv_path, additional_csv_path], dd.read_csv),
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
        dir_data_item = mlrun.run.get_dataitem(self._dbfs_schema + df_dir)
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
