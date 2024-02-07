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
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from databricks.sdk import WorkspaceClient

import mlrun
import mlrun.errors
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileDBFS,
    register_temporary_client_datastore_profile,
)
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
@pytest.mark.parametrize("use_datastore_profile", [False, True])
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
        self.profile_name = "dbfs_ds_profile"

        with open(self.test_file_path, "r") as f:
            self.test_string = f.read()

        self.profile = DatastoreProfileDBFS(
            name=self.profile_name,
            endpoint_url=env_params["DATABRICKS_HOST"],
            token=env_params["DATABRICKS_TOKEN"],
        )

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        setup_dbfs_dirs(
            workspace=self.workspace,
            specific_test_class_dir=self.dbfs_store_dir,
            subdirs=[self.parquets_dir, self.csv_dir],
        )
        register_temporary_client_datastore_profile(self.profile)
        store_manager.reset_secrets()
        env_params = self.config["env"]
        if use_datastore_profile:
            for key, env_param in env_params.items():
                os.environ[key] = "wrong_value"
        else:
            for key, env_param in env_params.items():
                os.environ[key] = env_param

    def teardown_class(self):
        teardown_dbfs_dirs(
            workspace=self.workspace, specific_test_class_dir=self.dbfs_store_dir
        )

    def _get_data_item(self, secrets={}, use_profile=False):
        object_path = f"{self.dbfs_store_path}/file_{uuid.uuid4()}.txt"
        object_url = (
            f"ds://{self.profile_name}{object_path}"
            if use_profile
            else f"{self._dbfs_schema}{object_path}"
        )
        return mlrun.run.get_dataitem(object_url, secrets=secrets), object_url

    @pytest.mark.parametrize("use_secrets_as_parameters", [True, False])
    def test_put_get_and_download(
        self, use_datastore_profile, use_secrets_as_parameters
    ):
        secrets = {}
        token = self.config["env"].get("DATABRICKS_TOKEN", None)
        host = self.config["env"].get("DATABRICKS_HOST", None)
        if use_secrets_as_parameters:
            os.environ["DATABRICKS_TOKEN"] = ""
            #  Verify that we are using the profile secret:
            secrets = (
                {"DATABRICKS_TOKEN": "wrong_token", "DATABRICKS_HOST": "wrong_host"}
                if use_datastore_profile
                else {"DATABRICKS_TOKEN": token, "DATABRICKS_HOST": host}
            )
        data_item, _ = self._get_data_item(
            secrets=secrets,
            use_profile=use_datastore_profile,
        )
        data_item.put(self.test_string)
        response = data_item.get()
        assert response.decode() == self.test_string
        response = data_item.get(offset=20)
        assert response.decode() == self.test_string[20:]
        response = data_item.get(size=20)
        assert response.decode() == self.test_string[:20]
        response = data_item.get(offset=20, size=0)
        assert response.decode() == self.test_string[20:]
        response = data_item.get(offset=20, size=10)
        assert response.decode() == self.test_string[20:30]

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == self.test_string

    def test_stat(self, use_datastore_profile):
        data_item, _ = self._get_data_item(use_profile=use_datastore_profile)
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    #
    def test_list_dir(self, use_datastore_profile):
        data_item, object_url = self._get_data_item(use_profile=use_datastore_profile)
        data_item.put(self.test_string)
        file_name_length = len(object_url.split("/")[-1]) + 1
        dir_dataitem = mlrun.run.get_dataitem(
            object_url[:-file_name_length],
        )
        dir_list = dir_dataitem.listdir()
        assert object_url.split("/")[-1] in dir_list

    def test_upload(self, use_datastore_profile):
        data_item, _ = self._get_data_item(use_profile=use_datastore_profile)
        data_item.upload(self.test_file_path)
        response = data_item.get()
        assert response.decode() == self.test_string

    #
    def test_rm(self, use_datastore_profile):
        data_item, _ = self._get_data_item(use_profile=use_datastore_profile)
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
    def test_as_df(
        self,
        use_datastore_profile,
        file_extension: str,
        local_file_path: str,
        reader: callable,
    ):
        source = reader(local_file_path)
        upload_file_path = (
            f"{self.dbfs_store_path}/file_{uuid.uuid4()}.{file_extension}"
        )
        dataitem_url = (
            f"ds://{self.profile_name}{upload_file_path}"
            if use_datastore_profile
            else self._dbfs_schema + upload_file_path
        )
        upload_data_item = mlrun.run.get_dataitem(dataitem_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df()
        assert source.equals(response)

    @pytest.mark.parametrize(
        "file_extension, local_file_path, reader, reader_args",
        [
            ("parquet", parquet_path, dd.read_parquet, {}),
            ("csv", csv_path, dd.read_csv, {}),
            ("json", json_path, dd.read_json, {"orient": "values"}),
        ],
    )
    def test_as_df_dd(
        self,
        use_datastore_profile,
        file_extension: str,
        local_file_path: str,
        reader: callable,
        reader_args: dict,
    ):
        if use_datastore_profile:
            pytest.skip(
                "dask dataframe is not supported by datastore profile."
            )  # TODO add support
        source = reader(local_file_path, **reader_args)
        upload_file_path = (
            f"{self.dbfs_store_path}/file_{uuid.uuid4()}.{file_extension}"
        )
        dataitem_url = (
            f"ds://{self.profile_name}{upload_file_path}"
            if use_datastore_profile
            else self._dbfs_schema + upload_file_path
        )
        upload_data_item = mlrun.run.get_dataitem(dataitem_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd, **reader_args)
        assert dd.assert_eq(source, response)

    def _setup_df_dir(
        self, use_profile, first_file_path, second_file_path, file_extension, directory
    ):
        upload_file_path = (
            f"{self.dbfs_store_path}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        dataitem_url = (
            f"ds://{self.profile_name}{upload_file_path}"
            if use_profile
            else self._dbfs_schema + upload_file_path
        )

        uploaded_data_item = mlrun.run.get_dataitem(dataitem_url)
        uploaded_data_item.upload(first_file_path)

        upload_file_path = (
            f"{self.dbfs_store_path}{directory}/file_{uuid.uuid4()}.{file_extension}"
        )
        dataitem_url = (
            f"ds://{self.profile_name}{upload_file_path}"
            if use_profile
            else self._dbfs_schema + upload_file_path
        )
        uploaded_data_item = mlrun.run.get_dataitem(dataitem_url)
        uploaded_data_item.upload(second_file_path)
        upload_directory = os.path.dirname(upload_file_path)
        return (
            f"ds://{self.profile_name}{upload_directory}"
            if use_profile
            else self._dbfs_schema + upload_directory
        )

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
        use_datastore_profile,
        directory: str,
        file_format: str,
        file_extension: str,
        files_paths: List[Path],
        reader: callable,
    ):
        first_file_path = files_paths[0]
        second_file_path = files_paths[1]
        dir_url = self._setup_df_dir(
            use_profile=use_datastore_profile,
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
            directory=directory,
        )

        dir_data_item = mlrun.run.get_dataitem(dir_url)
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
        use_datastore_profile,
        directory: str,
        file_format: str,
        file_extension: str,
        files_paths: List[Path],
        reader: callable,
    ):
        if use_datastore_profile:
            pytest.skip(
                "dask dataframe is not supported by datastore profile."
            )  # TODO add support
        first_file_path = files_paths[0]
        second_file_path = files_paths[1]
        df_url = self._setup_df_dir(
            use_profile=use_datastore_profile,
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
            directory=directory,
        )
        dir_data_item = mlrun.run.get_dataitem(df_url)
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

    def test_multiple_dataitems(self, use_datastore_profile):
        if not use_datastore_profile:
            pytest.skip("test_multiple_dataitems relevant for profiles only.")
        data_item, _ = self._get_data_item(use_profile=use_datastore_profile)
        test_profile = DatastoreProfileDBFS(
            name="test_profile",
            endpoint_url="test_host",
            token="test_token",
        )
        register_temporary_client_datastore_profile(test_profile)
        test_data_item = mlrun.run.get_dataitem(
            "ds://test_profile/test_directory/test_file.txt", secrets={}
        )
        assert data_item.store.to_dict() != test_data_item.store.to_dict()
