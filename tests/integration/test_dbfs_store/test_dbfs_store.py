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
import os.path
import tempfile
import uuid

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
        os.path.join(os.path.dirname(__file__), "test-dbfs-store.yml")
    ),
    reason="DBFS storage parameters not configured",
)
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestDBFSStore:
    class_dir = "/dbfs_store"
    test_dir = f"{MLRUN_ROOT_DIR}{class_dir}"
    run_dir = f"{test_dir}/run_{uuid.uuid4()}"
    profile_name = "dbfs_ds_profile"
    here = os.path.dirname(__file__)
    config_file_path = os.path.join(here, "test-dbfs-store.yml")
    assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    test_file = os.path.join(assets_path, "test.txt")

    @classmethod
    def setup_class(cls):
        with open(cls.config_file_path) as yaml_file:
            cls.config = yaml.safe_load(yaml_file)
        env_params = cls.config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        cls.token = env_params.get("DATABRICKS_TOKEN", None)
        cls.host = env_params.get("DATABRICKS_HOST", None)
        cls.workspace = WorkspaceClient()
        with open(cls.test_file) as f:
            cls.test_string = f.read()

        cls.profile = DatastoreProfileDBFS(
            name=cls.profile_name,
            endpoint_url=env_params["DATABRICKS_HOST"],
            token=env_params["DATABRICKS_TOKEN"],
        )
        setup_dbfs_dirs(
            workspace=cls.workspace,
            specific_test_class_dir=cls.class_dir,
            subdirs=[],
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
        self.object_url = f"{self.run_dir_url}{self.object_file}"
        register_temporary_client_datastore_profile(self.profile)
        store_manager.reset_secrets()

    @classmethod
    def teardown_class(cls):
        teardown_dbfs_dirs(
            workspace=cls.workspace, specific_test_class_dir=cls.class_dir
        )

    def teardown_method(self, method):
        os.environ["DATABRICKS_TOKEN"] = self.token
        os.environ["DATABRICKS_HOST"] = self.host

    @pytest.mark.parametrize("use_secrets_as_parameters", [True, False])
    def test_put_get_and_download(
        self, use_datastore_profile, use_secrets_as_parameters
    ):
        secrets = {}
        if use_secrets_as_parameters:
            os.environ["DATABRICKS_TOKEN"] = ""
            # Verify that we are using the correct profile secret by deliberately
            # setting an incorrect token as the secret or env.
            # We expect that the correct token, which is saved in the datastore profile, will be utilized.
            secrets = (
                {"DATABRICKS_TOKEN": "wrong_token", "DATABRICKS_HOST": "wrong_host"}
                if use_datastore_profile
                else {"DATABRICKS_TOKEN": self.token, "DATABRICKS_HOST": self.host}
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

    def test_stat(self):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    #
    def test_list_dir(self):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.put(self.test_string)
        file_name_length = len(self.object_url.split("/")[-1]) + 1
        dir_dataitem = mlrun.run.get_dataitem(
            self.object_url[:-file_name_length],
        )
        dir_list = dir_dataitem.listdir()
        assert self.object_url.split("/")[-1] in dir_list

    def test_upload(self):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.upload(self.test_file)
        response = data_item.get()
        assert response.decode() == self.test_string

    #
    def test_rm(self):
        data_item = mlrun.run.get_dataitem(self.object_url)
        data_item.upload(self.test_file)
        data_item.stat()
        data_item.delete()
        with pytest.raises(FileNotFoundError) as file_not_found_error:
            data_item.stat()
        assert "No file or directory exists on path" in str(file_not_found_error.value)

    @pytest.mark.parametrize(
        "file_format, pd_reader, dd_reader, reader_args",
        [
            ("parquet", pd.read_parquet, dd.read_parquet, {}),
            ("csv", pd.read_csv, dd.read_csv, {}),
            ("json", pd.read_json, dd.read_json, {"orient": "records"}),
        ],
    )
    def test_as_df(
        self,
        use_datastore_profile: bool,
        file_format: str,
        pd_reader: callable,
        dd_reader: callable,
        reader_args: dict,
    ):
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = os.path.join(self.assets_path, f"test_data.{file_format}")

        source = pd_reader(local_file_path, **reader_args)
        upload_data_item = mlrun.run.get_dataitem(dataframe_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(**reader_args)
        pd.testing.assert_frame_equal(source, response)

        # dask
        source = dd_reader(local_file_path, **reader_args)
        response = upload_data_item.as_df(**reader_args, df_module=dd)
        dd.assert_eq(source, response)

    @pytest.mark.parametrize(
        "file_format, pd_reader, dd_reader, reset_index",
        [
            ("parquet", pd.read_parquet, dd.read_parquet, False),
            ("csv", pd.read_csv, dd.read_csv, True),
        ],
    )
    def test_as_df_directory(self, file_format, pd_reader, dd_reader, reset_index):
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = os.path.join(self.assets_path, f"test_data.{file_format}")
        df2_path = os.path.join(self.assets_path, f"additional_data.{file_format}")

        # upload
        dt1 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df1.{file_format}",
        )
        dt2 = mlrun.run.get_dataitem(f"{dataframes_url}/df2.{file_format}")
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        dt_dir = mlrun.run.get_dataitem(dataframes_url)
        df1 = pd_reader(df1_path)
        df2 = pd_reader(df2_path)
        expected_df = pd.concat([df1, df2], ignore_index=True)
        tested_df = dt_dir.as_df(format=file_format)
        if reset_index:
            tested_df = tested_df.sort_values("ID").reset_index(drop=True)
        pd.testing.assert_frame_equal(tested_df, expected_df)

        # dask
        dd_df1 = dd_reader(df1_path)
        dd_df2 = dd_reader(df2_path)
        expected_dd_df = dd.concat([dd_df1, dd_df2], axis=0)
        tested_dd_df = dt_dir.as_df(format=file_format, df_module=dd)
        dd.assert_eq(tested_dd_df, expected_dd_df)

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
