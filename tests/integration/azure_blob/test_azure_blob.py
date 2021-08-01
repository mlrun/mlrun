import os
import random
from pathlib import Path

import pytest
import yaml

import mlrun

here = Path(__file__).absolute().parent
config_file_path = here / "test-azure-blob.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename, "r") as f:
    test_string = f.read()

BLOB_DIR = "test_mlrun_azure_blob"
BLOB_FILE = f"file_{random.randint(0, 1000)}.txt"

AUTH_METHODS_AND_REQUIRED_PARAMS = {
    "conn_str": ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_CONTAINER"],
    "sas_token": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_SAS_TOKEN", "AZURE_SAS_CONTAINER"],
    "account_key": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY", "AZURE_KEY_CONTAINER"],
    "spn": [
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_STORAGE_TENANT_ID",
        "AZURE_SPN_CONTAINER",
    ],
}


def azure_connection_configured():
    return config["env"].get("AZURE_STORAGE_CONNECTION_STRING") is not None

@pytest.fixture
def auth_method(request):
    # Remove any previously existing env_vars needed for authentication
    print(f"setup {request.param}")
    for k, env_vars in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
        for env_var in env_vars:
            os.environ.pop(env_var, None)
    env_vars = AUTH_METHODS_AND_REQUIRED_PARAMS.get(request.param)
    for env_var in env_vars:
        if "CONTAINER" in env_var:
            blob_path = "az://" + config["env"].get(env_var)
        else:
            os.environ[env_var] = config["env"].get(env_var)

    blob_url = blob_path + "/" + BLOB_DIR + "/" + BLOB_FILE
    print(f"\nBlob URL: {blob_url}")
    data_item = mlrun.run.get_dataitem(blob_url)
    data_item.put(test_string)
    yield data_item, request.param
    print('teardown')


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason="This is an integration test, add the needed environment variables in test-azure-blob.yml "
    "to run it",
)
@pytest.mark.parametrize("auth_method", ["conn_str", "account_key", "sas_token", "spn"], indirect=["auth_method"])
def test_azure_blob(auth_method):
    data_item, auth = auth_method[0], auth_method[1]
    print(auth)
    response = data_item.get()
    assert response.decode() == test_string, "Result differs from original test"

    response = data_item.get(offset=20)
    assert response.decode() == test_string[20:], "Partial result not as expected"

    stat = data_item.stat()
    assert stat.size == len(test_string), "Stat size different than expected"


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason="This is an integration test, add the needed environment variables in test-azure-blob.yml "
    "to run it",
)
@pytest.mark.parametrize("auth_method", 
                         ["conn_str", "account_key",  "sas_token", "spn"],
                         indirect=["auth_method"]
                         )
def test_list_dir(auth_method):
    _, auth = auth_method[0], auth_method[1]
    env_vars = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth)
    for env_var in env_vars:
        if "CONTAINER" in env_var:
            blob_container_path = "az://" + config["env"].get(env_var)
    
    dir_list = mlrun.run.get_dataitem(blob_container_path).listdir()
    
    # # Check dir list for container
    assert BLOB_DIR + "/" + BLOB_FILE in dir_list, "File not in container dir-list"

    # Check dir list for folder in container
    dir_list = mlrun.run.get_dataitem(
        blob_container_path + "/" + BLOB_DIR
    ).listdir()
    assert BLOB_FILE in dir_list, "File not in folder dir-list"


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason="This is an integration test, add the needed environment variables in test-azure-blob.yml "
    "to run it",
)
@pytest.mark.parametrize("auth_method", 
                         ["conn_str", "account_key",  "sas_token", "spn"],
                         indirect=["auth_method"]
                         )
def test_blob_upload(auth_method):
    _, auth = auth_method[0], auth_method[1]
    env_vars = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth)
    for env_var in env_vars:
        if "CONTAINER" in env_var:
            blob_path = "az://" + config["env"].get(env_var)

    blob_url = blob_path + "/" + BLOB_DIR + "/" + BLOB_FILE
    print(f"\nBlob URL: {blob_url}")
    
    upload_data_item = mlrun.run.get_dataitem(blob_url)
    upload_data_item.upload(test_filename)

    response = upload_data_item.get()
    assert response.decode() == test_string, "Result differs from original test"
