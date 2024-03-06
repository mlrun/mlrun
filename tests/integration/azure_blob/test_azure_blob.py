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
# @pytest.mark.parametrize("auth_method", list(AUTH_METHODS_AND_REQUIRED_PARAMS.keys()))
@pytest.mark.parametrize(
    "auth_method ,use_datastore_profile", generated_pytest_parameters
)
@pytest.mark.skipif(
    not config["env"].get("AZURE_CONTAINER"),
    reason="AZURE_CONTAINER is not set",
)
class TestAzureBlob:
    @classmethod
    def setup_class(cls):
        cls.profile_name = "azure_blob_ds_profile"
        cls.test_dir = "test_mlrun_azure_blob"
        cls._bucket_name = config["env"].get("AZURE_CONTAINER", None)
        cls.test_file = here / "test.txt"
        with open(cls.test_file, "r") as f:
            cls.test_string = f.read()

    def teardown_method(self):
        test_dir = f"{self._bucket_name}/{self.test_dir}"
        if not self._azure_fs:
            return
        if self._azure_fs.exists(test_dir):
            self._azure_fs.delete(test_dir, recursive=True)
            logger.debug("test directory has been cleaned.")

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile, auth_method):
        self.blob_file = f"file_{uuid.uuid4()}.txt"
        store_manager.reset_secrets()
        self._azure_fs = None
        self.storage_options = {}
        for k, env_vars in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
            for env_var in env_vars:
                os.environ.pop(env_var, None)

        test_params = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth_method)
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
            self._azure_fs = AzureBlobFileSystem()
        elif auth_method.startswith("fsspec"):
            for var in test_params:
                value = config["env"].get(var)
                if not value:
                    pytest.skip(f"Auth method {auth_method} not configured.")
                self.storage_options[var] = value
            self._azure_fs = AzureBlobFileSystem(**self.storage_options)
            logger.info(f"Testing auth method {auth_method}")
            if use_datastore_profile:
                self.profile = DatastoreProfileAzureBlob(
                    name=self.profile_name, **self.storage_options
                )
                register_temporary_client_datastore_profile(self.profile)
        else:
            raise ValueError("auth_method not known")

    def get_blob_container_path(self, use_datastore_profile):
        if use_datastore_profile:
            return f"ds://{self.profile_name}/{self._bucket_name}"
        return "az://" + self._bucket_name

    def test_azure_blob(self, use_datastore_profile, auth_method):
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file
        data_item = mlrun.run.get_dataitem(blob_url, secrets=self.storage_options)
        data_item.put(self.test_string)

        # Validate append is properly blocked (currently not supported for Azure blobs)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            data_item.put("just checking!", append=True)

        response = data_item.get()
        assert (
            response.decode() == self.test_string
        ), "Result differs from original test"

        response = data_item.get(offset=20)
        assert (
            response.decode() == self.test_string[20:]
        ), "Partial result not as expected"
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == self.test_string

        stat = data_item.stat()
        assert stat.size == len(self.test_string), "Stat size different than expected"

    def test_list_dir(self, use_datastore_profile, auth_method):
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file

        file_dataitem = mlrun.run.get_dataitem(blob_url, self.storage_options)
        file_dataitem.put(self.test_string)

        # Check dir list for container
        blob_item = mlrun.run.get_dataitem(blob_container_path, self.storage_options)
        dir_list = blob_item.listdir()  # can take a lot of time to big buckets.
        assert (
            self.test_dir + "/" + self.blob_file in dir_list
        ), "File not in container dir-list"

        # Check dir list for folder in container
        dir_dataitem = mlrun.run.get_dataitem(
            blob_container_path + "/" + self.test_dir, self.storage_options
        )
        assert self.blob_file in dir_dataitem.listdir(), "File not in folder dir-list"
        file_dataitem.delete()
        assert self.blob_file not in dir_dataitem.listdir()

    def test_blob_upload(self, use_datastore_profile, auth_method):
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file

        upload_data_item = mlrun.run.get_dataitem(blob_url, self.storage_options)
        upload_data_item.upload(self.test_file)

        response = upload_data_item.get()
        assert (
            response.decode() == self.test_string
        ), "Result differs from original test"

    @pytest.mark.parametrize(
        "file_format, file_extension, ,writer, reader",
        [
            (
                "parquet",
                "parquet",
                pd.DataFrame.to_parquet,
                pd.read_parquet,
            ),
            ("csv", "csv", pd.DataFrame.to_csv, pd.read_csv),
            ("json", "json", pd.DataFrame.to_json, pd.read_json),
        ],
    )
    def test_as_df(
        self,
        use_datastore_profile,
        auth_method,
        file_format: str,
        file_extension: str,
        writer: callable,
        reader: callable,
    ):
        data = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}", delete=True
        ) as temp_file:
            writer_kwargs = {"index": False} if file_format != "json" else {}
            writer(df, temp_file.name, **writer_kwargs)
            blob_container_path = self.get_blob_container_path(use_datastore_profile)
            blob_url = (
                blob_container_path
                + "/"
                + self.test_dir
                + "/"
                + f"file{uuid.uuid4()}.{file_extension}"
            )
            upload_data_item = mlrun.run.get_dataitem(blob_url, self.storage_options)
            upload_data_item.upload(temp_file.name)

            result_df = upload_data_item.as_df()
            assert result_df.equals(df)

    @pytest.mark.parametrize(
        "directory, file_format, file_extension, ,writer, reader",
        [
            (
                parquets_dir,
                "parquet",
                "parquet",
                pd.DataFrame.to_parquet,
                pd.read_parquet,
            ),
            (csv_dir, "csv", "csv", pd.DataFrame.to_csv, pd.read_csv),
        ],
    )
    def test_read_df_dir(
        self,
        use_datastore_profile,
        auth_method,
        directory: str,
        file_format: str,
        file_extension: str,
        writer: callable,
        reader: callable,
    ):
        #  generate dfs
        # Define data for the first DataFrame
        data1 = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}

        # Define data for the second DataFrame
        data2 = {"Column1": [4, 5, 6], "Column2": ["X", "Y", "Z"]}

        # Create the DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        with (
            tempfile.NamedTemporaryFile(
                suffix=f".{file_extension}", delete=True
            ) as temp_file1,
            tempfile.NamedTemporaryFile(
                suffix=f".{file_extension}", delete=True
            ) as temp_file2,
        ):
            first_file_path = temp_file1.name
            second_file_path = temp_file2.name
            writer(df1, temp_file1.name, index=False)
            writer(df2, temp_file2.name, index=False)

            blob_container_path = self.get_blob_container_path(use_datastore_profile)
            dir_url = (
                f"{blob_container_path}/{self.test_dir}/{directory}/{uuid.uuid4()}"
            )
            first_file_url = f"{dir_url}/first_file.{file_extension}"
            second_file_url = f"{dir_url}/second_file.{file_extension}"
            first_file_data_item = mlrun.run.get_dataitem(
                first_file_url, self.storage_options
            )
            second_file_data_item = mlrun.run.get_dataitem(
                second_file_url, self.storage_options
            )
            first_file_data_item.upload(first_file_path)
            second_file_data_item.upload(second_file_path)

            #  start the test:
            dir_data_item = mlrun.run.get_dataitem(dir_url, self.storage_options)
            response_df = (
                dir_data_item.as_df(format=file_format)
                .sort_values("Column1")
                .reset_index(drop=True)
            )
            appended_df = (
                pd.concat([df1, df2], axis=0)
                .sort_values("Column1")
                .reset_index(drop=True)
            )
            assert response_df.equals(appended_df)
