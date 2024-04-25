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
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from adlfs.spec import AzureBlobFileSystem

import mlrun
import mlrun.errors
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileAzureBlob,
    register_temporary_client_datastore_profile,
)
from mlrun.utils import logger

here = Path(__file__).absolute().parent

parquets_dir = "parquets"
csv_dir = "csv"

config_file_path = here / "test-azure-blob.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

AUTH_METHODS_AND_REQUIRED_PARAMS = {
    "env_conn_str": ["AZURE_STORAGE_CONNECTION_STRING"],
    "env_sas_token": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_SAS_TOKEN"],
    "env_account_key": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"],
    "env_spn": [
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_STORAGE_TENANT_ID",
    ],
    "fsspec_conn_str": ["connection_string"],
    "fsspec_sas_token": ["account_name", "sas_token"],
    "fsspec_account_key": ["account_name", "account_key"],
    "fsspec_spn": ["account_name", "client_id", "client_secret", "tenant_id"],
    "fsspec_credential": ["credential"],
}

generated_pytest_parameters = []
for authentication_method in AUTH_METHODS_AND_REQUIRED_PARAMS:
    generated_pytest_parameters.append((authentication_method, False))
    if authentication_method.startswith("fsspec"):
        generated_pytest_parameters.append((authentication_method, True))


# Apply parametrization to all tests in this file. Skip test if auth method is not configured.
@pytest.mark.parametrize(
    "auth_method ,use_datastore_profile", generated_pytest_parameters
)
@pytest.mark.skipif(
    not config["env"].get("AZURE_CONTAINER"),
    reason="AZURE_CONTAINER is not set",
)
class TestAzureBlob:
    assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    profile_name = "azure_blob_ds_profile"
    test_dir = "test_mlrun_azure_blob"
    run_dir = f"{test_dir}/run_{uuid.uuid4()}"
    bucket_name = config["env"].get("AZURE_CONTAINER", None)
    test_file = os.path.join(assets_path, "test.txt")

    @classmethod
    def setup_class(cls):
        with open(cls.test_file) as f:
            cls.test_string = f.read()
        cls._azure_fs = None

    @classmethod
    def teardown_class(cls):
        test_dir = f"{cls.bucket_name}/{cls.test_dir}"
        if not cls._azure_fs:
            return
        if cls._azure_fs.exists(test_dir):
            cls._azure_fs.delete(test_dir, recursive=True)
            logger.debug("test directory has been deleted.")

    def teardown_method(self, method):
        for auth, auth_list in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
            if auth.startswith("env"):
                for env_parameter in auth_list:
                    if config["env"].get(env_parameter, None):
                        os.environ[env_parameter] = config["env"].get(env_parameter)

    @classmethod
    def create_fs(cls, storage_options):
        # Create filesystem object only once
        if not cls._azure_fs:
            azure_fs = AzureBlobFileSystem(storage_options)
            azure_fs.info(cls.bucket_name)  # in order to check connection ...
            cls._azure_fs = azure_fs

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile, auth_method):
        self.object_file = f"/file_{uuid.uuid4()}.txt"
        store_manager.reset_secrets()
        self.storage_options = {}
        for k, env_vars in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
            for env_var in env_vars:
                os.environ.pop(env_var, None)

        test_params = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth_method)
        if use_datastore_profile:
            self._bucket_url = f"ds://{self.profile_name}/{self.bucket_name}"
        else:
            self._bucket_url = f"az://{self.bucket_name}"
        self.run_dir_url = f"{self._bucket_url}/{self.run_dir}"
        self.object_url = f"{self.run_dir_url}{self.object_file}"

        if not test_params:
            pytest.skip(f"Auth method {auth_method} not configured.")

        if auth_method.startswith("env"):
            if use_datastore_profile:
                raise ValueError(
                    f"Auth method {auth_method} does not support profiles."
                )
            for env_var in test_params:
                env_value = config["env"].get(env_var)
                if not env_value:
                    pytest.skip(f"Auth method {auth_method} not configured.")
                os.environ[env_var] = env_value

            logger.info(f"Testing auth method {auth_method}")
        elif auth_method.startswith("fsspec"):
            for var in test_params:
                value = config["env"].get(var)
                if not value:
                    pytest.skip(f"Auth method {auth_method} not configured.")
                self.storage_options[var] = value
            logger.info(f"Testing auth method {auth_method}")
            if use_datastore_profile:
                self.profile = DatastoreProfileAzureBlob(
                    name=self.profile_name, **self.storage_options
                )
                register_temporary_client_datastore_profile(self.profile)
        else:
            raise ValueError("auth_method not known")
        self.create_fs(storage_options=self.storage_options)

    def test_azure_blob(self):
        data_item = mlrun.run.get_dataitem(
            self.object_url, secrets=self.storage_options
        )
        data_item.put(self.test_string)

        # Validate append is properly blocked (currently not supported for Azure blobs)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            data_item.put("just checking!", append=True)

        response = data_item.get()
        assert response.decode() == self.test_string

        response = data_item.get(offset=20)
        assert response.decode() == self.test_string[20:]
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == self.test_string

        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    def test_list_dir(self):
        file_dataitem = mlrun.run.get_dataitem(self.object_url, self.storage_options)
        file_dataitem.put(self.test_string)

        # Check dir list for container
        blob_item = mlrun.run.get_dataitem(self._bucket_url, self.storage_options)
        dir_list = blob_item.listdir()  # can take a lot of time to big buckets.
        assert f"{self.run_dir}{self.object_file}" in dir_list

        # Check dir list for folder in container
        dir_dataitem = mlrun.run.get_dataitem(self.run_dir_url, self.storage_options)
        assert self.object_file.split("/")[-1] in dir_dataitem.listdir()
        file_dataitem.delete()
        assert self.object_file.split("/")[-1] not in dir_dataitem.listdir()

    def test_blob_upload(self):
        upload_data_item = mlrun.run.get_dataitem(self.object_url, self.storage_options)
        upload_data_item.upload(self.test_file)

        response = upload_data_item.get()
        assert response.decode() == self.test_string

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
        file_format: str,
        pd_reader: callable,
        dd_reader: callable,
        reader_args: dict,
    ):
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = os.path.join(self.assets_path, f"test_data.{file_format}")

        source = pd_reader(local_file_path, **reader_args)
        upload_data_item = mlrun.run.get_dataitem(
            dataframe_url, secrets=self.storage_options
        )
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
    def test_as_df_directory(
        self,
        file_format,
        pd_reader,
        dd_reader,
        reset_index,
    ):
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = os.path.join(self.assets_path, f"test_data.{file_format}")
        df2_path = os.path.join(self.assets_path, f"additional_data.{file_format}")

        # upload
        dt1 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df1.{file_format}", secrets=self.storage_options
        )
        dt2 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df2.{file_format}", secrets=self.storage_options
        )
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        dt_dir = mlrun.run.get_dataitem(dataframes_url, secrets=self.storage_options)
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
