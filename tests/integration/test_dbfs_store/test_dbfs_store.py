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
from os.path import abspath, dirname, join
from pathlib import Path

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

    @classmethod
    def setup_class(cls):
        cls.assets_path = join(dirname(dirname(abspath(__file__))), "assets")
        env_params = cls.config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        cls.test_dir = "/dbfs_store"
        cls.run_dir = f"{MLRUN_ROOT_DIR}{cls.test_dir}"
        cls.workspace = WorkspaceClient()
        cls.profile_name = "dbfs_ds_profile"

        cls.test_file = join(cls.assets_path, "test.txt")
        with open(cls.test_file) as f:
            cls.test_string = f.read()

        cls.profile = DatastoreProfileDBFS(
            name=cls.profile_name,
            endpoint_url=env_params["DATABRICKS_HOST"],
            token=env_params["DATABRICKS_TOKEN"],
        )

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self.object_file = f"/file_{uuid.uuid4()}.txt"
        env_params = self.config["env"]
        if use_datastore_profile:
            self.prefix_url = f"ds://{self.profile_name}"
            for key, env_param in env_params.items():
                os.environ[key] = "wrong_value"
        else:
            self.prefix_url = "dbfs://"
            for key, env_param in env_params.items():
                os.environ[key] = env_param
        self.run_dir_url = f"{self.prefix_url}{self.run_dir}"
        self.object_url = self.run_dir_url + self.object_file
        setup_dbfs_dirs(
            workspace=self.workspace,
            specific_test_class_dir=self.test_dir,
            subdirs=[],
        )
        register_temporary_client_datastore_profile(self.profile)
        store_manager.reset_secrets()

    @classmethod
    def teardown_class(cls):
        teardown_dbfs_dirs(
            workspace=cls.workspace, specific_test_class_dir=cls.test_dir
        )

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
        data_item = mlrun.run.get_dataitem(self.object_url, secrets=secrets)
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
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    #
    def test_list_dir(self, use_datastore_profile):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.put(self.test_string)
        file_name_length = len(self.object_url.split("/")[-1]) + 1
        dir_dataitem = mlrun.run.get_dataitem(
            self.object_url[:-file_name_length],
        )
        dir_list = dir_dataitem.listdir()
        assert self.object_url.split("/")[-1] in dir_list

    def test_upload(self, use_datastore_profile):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.upload(self.test_file)
        response = data_item.get()
        assert response.decode() == self.test_string

    #
    def test_rm(self, use_datastore_profile):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.upload(self.test_file)
        data_item.stat()
        data_item.delete()
        with pytest.raises(FileNotFoundError) as file_not_found_error:
            data_item.stat()
        assert "No file or directory exists on path" in str(file_not_found_error.value)

    @pytest.mark.parametrize(
        "file_format, reader, reader_args",
        [
            ("parquet", pd.read_parquet, {}),
            ("csv", pd.read_csv, {}),
            ("json", pd.read_json, {"orient": "values"}),
        ],
    )
    def test_as_df(
        self,
        use_datastore_profile,
        file_format: str,
        reader: callable,
        reader_args: dict,
    ):
        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path)
        dataframe_url = f"{self.run_dir_url}/file_{uuid.uuid4()}.{file_format}"
        upload_data_item = mlrun.run.get_dataitem(dataframe_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df()
        pd.testing.assert_frame_equal(source, response)

    @pytest.mark.parametrize(
        "file_format, reader, reader_args",
        [
            ("parquet", dd.read_parquet, {}),
            ("csv", dd.read_csv, {}),
            ("json", dd.read_json, {"orient": "values"}),
        ],
    )
    def test_as_df_dd(
        self,
        use_datastore_profile,
        file_format: str,
        reader: callable,
        reader_args: dict,
    ):
        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)
        dataframe_url = f"{self.run_dir_url}/file_{uuid.uuid4()}.{file_format}"
        upload_data_item = mlrun.run.get_dataitem(dataframe_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd, **reader_args)
        assert dd.assert_eq(source, response)

    def _setup_df_dir(self, use_datastore_profile, file_format, reader):
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = join(self.assets_path, f"test_data.{file_format}")
        df2_path = join(self.assets_path, f"additional_data.{file_format}")

        #  upload
        dt1 = mlrun.run.get_dataitem(dataframes_url + f"/df1.{file_format}")
        dt2 = mlrun.run.get_dataitem(dataframes_url + f"/df2.{file_format}")
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        return (
            mlrun.run.get_dataitem(dataframes_url),
            reader(df1_path),
            reader(df2_path),
        )

    @pytest.mark.parametrize(
        "file_format, reader, reset_index",
        [
            ("parquet", pd.read_parquet, False),
            ("csv", pd.read_csv, True),
        ],
    )
    def test_as_df_directory(
        self, use_datastore_profile, file_format, reader, reset_index
    ):
        dt_dir, df1, df2 = self._setup_df_dir(
            use_datastore_profile=use_datastore_profile,
            file_format=file_format,
            reader=reader,
        )
        tested_df = dt_dir.as_df(format=file_format)
        if reset_index:
            tested_df = tested_df.sort_values("ID").reset_index(drop=True)
        expected_df = pd.concat([df1, df2], ignore_index=True)
        pd.testing.assert_frame_equal(tested_df, expected_df)

    @pytest.mark.parametrize(
        "file_format, reader, reset_index",
        [
            ("parquet", dd.read_parquet, False),
            ("csv", dd.read_csv, True),
        ],
    )
    def test_as_df_directory_dd(
        self, use_datastore_profile, file_format, reader, reset_index
    ):
        dt_dir, df1, df2 = self._setup_df_dir(
            use_datastore_profile=use_datastore_profile,
            file_format=file_format,
            reader=reader,
        )
        tested_df = dt_dir.as_df(format=file_format, df_module=dd)
        if reset_index:
            tested_df = tested_df.sort_values("ID").reset_index(drop=True)
        expected_df = dd.concat([df1, df2], axis=0)
        dd.assert_eq(tested_df, expected_df)

    def test_multiple_dataitems(self, use_datastore_profile):
        if not use_datastore_profile:
            pytest.skip("test_multiple_dataitems relevant for profiles only.")
        data_item = mlrun.run.get_dataitem(self.object_url)
        test_profile = DatastoreProfileDBFS(
            name="test_profile",
            endpoint_url="test_host",
            token="test_token",
        )
        register_temporary_client_datastore_profile(test_profile)
        test_data_item = mlrun.run.get_dataitem(
            "ds://test_profile/test_directory/test_file.txt", secrets={}
        )
        assert data_item.store.to_dict() != test_data_item._store.to_dict()
