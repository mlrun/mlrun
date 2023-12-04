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
    # if authentication_method.startswith("fsspec"):
    #     generated_pytest_parameters.append((authentication_method, True))


# Apply parametrization to all tests in this file. Skip test if auth method is not configured.
# @pytest.mark.parametrize("auth_method", list(AUTH_METHODS_AND_REQUIRED_PARAMS.keys()))
@pytest.mark.parametrize(
    "auth_method ,use_datastore_profile", generated_pytest_parameters
)
class TestAzureBlob:
    @classmethod
    def setup_class(cls):
        cls.profile_name = "azure_blob_ds_profile"
        cls.test_dir = "test_mlrun_azure_blob"

        config_file_path = here / "test-azure-blob.yml"
        with config_file_path.open() as fp:
            cls.config = yaml.safe_load(fp)

        cls.test_file = here / "test.txt"
        with open(cls.test_file, "r") as f:
            cls.test_string = f.read()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self.blob_file = f"file_{uuid.uuid4()}.txt"
        store_manager.reset_secrets()

    def get_blob_container_path(self, use_datastore_profile):
        if use_datastore_profile:
            return (
                f"ds://{self.profile_name}/{self.config['env'].get('AZURE_CONTAINER')}"
            )
        return "az://" + self.config["env"].get("AZURE_CONTAINER")

    def verify_auth_parameters_and_configure(self, auth_method, use_datastore_profile):
        # This sets up the authentication method against Azure
        # if testing the use of Azure credentials stored as
        # environmental variable, it creates the environmental
        # variables and returns storage_options = None.  Otherwise
        # it returns adlfs-recognized parameters compliant with the
        # fsspec api.  These get saved as secrets by mlrun.get_dataitem()
        # for authentication.
        if not self.config["env"].get("AZURE_CONTAINER"):
            pytest.skip(f"Auth method {auth_method} not configured.")

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
                env_value = self.config["env"].get(env_var)
                if not env_value:
                    pytest.skip(f"Auth method {auth_method} not configured.")
                os.environ[env_var] = env_value

            logger.info(f"Testing auth method {auth_method}")
            return {}

        elif auth_method.startswith("fsspec"):
            storage_options = {}
            for var in test_params:
                value = self.config["env"].get(var)
                if not value:
                    pytest.skip(f"Auth method {auth_method} not configured.")
                storage_options[var] = value
            logger.info(f"Testing auth method {auth_method}")
            if use_datastore_profile:
                self.profile = DatastoreProfileAzureBlob(
                    name=self.profile_name, **storage_options
                )
                register_temporary_client_datastore_profile(self.profile)
            else:
                return storage_options

        else:
            raise ValueError("auth_method not known")

    def test_azure_blob(self, use_datastore_profile, auth_method):
        storage_options = self.verify_auth_parameters_and_configure(
            auth_method, use_datastore_profile
        )
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file

        print(f"\nBlob URL: {blob_url}")

        data_item = mlrun.run.get_dataitem(blob_url, secrets=storage_options)
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

        stat = data_item.stat()
        assert stat.size == len(self.test_string), "Stat size different than expected"

    def test_list_dir(self, use_datastore_profile, auth_method):
        storage_options = self.verify_auth_parameters_and_configure(
            auth_method, use_datastore_profile
        )
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file
        print(f"\nBlob URL: {blob_url}")

        mlrun.run.get_dataitem(blob_url, storage_options).put(self.test_string)

        # Check dir list for container
        dir_item = mlrun.run.get_dataitem(blob_container_path, storage_options)
        dir_list = dir_item.listdir()  # can take a lot of time to big buckets.
        assert (
            self.test_dir + "/" + self.blob_file in dir_list
        ), "File not in container dir-list"

        # Check dir list for folder in container
        dir_list = mlrun.run.get_dataitem(
            blob_container_path + "/" + self.test_dir, storage_options
        ).listdir()
        assert self.blob_file in dir_list, "File not in folder dir-list"

    def test_blob_upload(self, use_datastore_profile, auth_method):
        storage_options = self.verify_auth_parameters_and_configure(
            auth_method, use_datastore_profile
        )
        blob_container_path = self.get_blob_container_path(use_datastore_profile)
        blob_url = blob_container_path + "/" + self.test_dir + "/" + self.blob_file
        print(f"\nBlob URL: {blob_url}")

        upload_data_item = mlrun.run.get_dataitem(blob_url, storage_options)
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
        storage_options = self.verify_auth_parameters_and_configure(
            auth_method, use_datastore_profile
        )
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
            upload_data_item = mlrun.run.get_dataitem(blob_url, storage_options)
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
        storage_options = self.verify_auth_parameters_and_configure(
            auth_method, use_datastore_profile
        )
        #  generate dfs
        # Define data for the first DataFrame
        data1 = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}

        # Define data for the second DataFrame
        data2 = {"Column1": [4, 5, 6], "Column2": ["X", "Y", "Z"]}

        # Create the DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}", delete=True
        ) as temp_file1, tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}", delete=True
        ) as temp_file2:
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
                first_file_url, storage_options
            )
            second_file_data_item = mlrun.run.get_dataitem(
                second_file_url, storage_options
            )
            first_file_data_item.upload(first_file_path)
            second_file_data_item.upload(second_file_path)

            #  start the test:
            dir_data_item = mlrun.run.get_dataitem(dir_url, storage_options)
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
