# Copyright 2018 Iguazio
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
import random
from pathlib import Path

import pytest
import yaml

import mlrun
import mlrun.errors
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-azure-blob.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename, "r") as f:
    test_string = f.read()

blob_dir = "test_mlrun_azure_blob"
blob_file = f"file_{random.randint(0, 1000)}.txt"

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


def verify_auth_parameters_and_configure_env(auth_method):
    # This sets up the authentication method against Azure
    # if testing the use of Azure credentials stored as
    # environmental variable, it creates the environmental
    # variables and returns storage_options = None.  Otherwise
    # it returns adlfs-recognized parameters compliant with the
    # fsspec api.  These get saved as secrets by mlrun.get_dataitem()
    # for authentication.
    if not config["env"].get("AZURE_CONTAINER"):
        return None

    for k, env_vars in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
        for env_var in env_vars:
            os.environ.pop(env_var, None)

    test_params = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth_method)
    if not test_params:
        return None

    if auth_method.startswith("env"):
        for env_var in test_params:
            env_value = config["env"].get(env_var)
            if not env_value:
                return None
            os.environ[env_var] = env_value

        logger.info(f"Testing auth method {auth_method}")
        return {}

    elif auth_method.startswith("fsspec"):
        storage_options = {}
        for var in test_params:
            value = config["env"].get(var)
            if not value:
                return None
            storage_options[var] = value
        logger.info(f"Testing auth method {auth_method}")
        return storage_options

    else:
        raise ValueError("auth_method not known")


# Apply parametrization to all tests in this file. Skip test if auth method is not configured.
pytestmark = pytest.mark.parametrize(
    "auth_method",
    [
        pytest.param(
            auth_method,
            marks=pytest.mark.skipif(
                verify_auth_parameters_and_configure_env(auth_method) is None,
                reason=f"Auth method {auth_method} not configured.",
            ),
        )
        for auth_method in AUTH_METHODS_AND_REQUIRED_PARAMS
    ],
)


def test_azure_blob(auth_method):
    storage_options = verify_auth_parameters_and_configure_env(auth_method)
    blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
    blob_url = blob_path + "/" + blob_dir + "/" + blob_file

    print(f"\nBlob URL: {blob_url}")

    data_item = mlrun.run.get_dataitem(blob_url, secrets=storage_options)
    data_item.put(test_string)

    # Validate append is properly blocked (currently not supported for Azure blobs)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        data_item.put("just checking!", append=True)

    response = data_item.get()
    assert response.decode() == test_string, "Result differs from original test"

    response = data_item.get(offset=20)
    assert response.decode() == test_string[20:], "Partial result not as expected"

    stat = data_item.stat()
    assert stat.size == len(test_string), "Stat size different than expected"


def test_list_dir(auth_method):
    storage_options = verify_auth_parameters_and_configure_env(auth_method)
    blob_container_path = "az://" + config["env"].get("AZURE_CONTAINER")
    blob_url = blob_container_path + "/" + blob_dir + "/" + blob_file
    print(f"\nBlob URL: {blob_url}")

    mlrun.run.get_dataitem(blob_url, storage_options).put(test_string)

    # Check dir list for container
    dir_list = mlrun.run.get_dataitem(blob_container_path, storage_options).listdir()
    assert blob_dir + "/" + blob_file in dir_list, "File not in container dir-list"

    # Check dir list for folder in container
    dir_list = mlrun.run.get_dataitem(
        blob_container_path + "/" + blob_dir, storage_options
    ).listdir()
    assert blob_file in dir_list, "File not in folder dir-list"


def test_blob_upload(auth_method):
    storage_options = verify_auth_parameters_and_configure_env(auth_method)
    blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
    blob_url = blob_path + "/" + blob_dir + "/" + blob_file
    print(f"\nBlob URL: {blob_url}")

    upload_data_item = mlrun.run.get_dataitem(blob_url, storage_options)
    upload_data_item.upload(test_filename)

    response = upload_data_item.get()
    assert response.decode() == test_string, "Result differs from original test"
