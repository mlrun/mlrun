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

blob_dir = "test_mlrun_azure_blob"
blob_file = f"file_{random.randint(0, 1000)}.txt"

AUTH_METHODS_AND_REQUIRED_PARAMS = {
    "conn_str": [
        "AZURE_STORAGE_CONNECTION_STRING"
    ],
    "sas_token": [
        "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_SAS_TOKEN"
    ],
    "account_key": [
        "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"
    ],
    "spn": [
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_STORAGE_TENANT_ID",
    ]
}


def azure_connection_configured():
    for var in AUTH_VARS:
        assert config["env"].get(var) is not None
    return True


def prepare_env(auth_method):
    for v in AUTH_VARS:
        os.environ.pop(v, None)
    if auth_method == "conn_str":
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = config["env"].get(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
    else:
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = config["env"].get(
            "AZURE_STORAGE_ACCOUNT_NAME"
        )

        if auth_method == "sas_token":
            os.environ["AZURE_STORAGE_SAS_TOKEN"] = config["env"].get(
                "AZURE_STORAGE_SAS_TOKEN"
            )
        elif auth_method == "account_key":
            os.environ["AZURE_STORAGE_ACCOUNT_KEY"] = config["env"].get(
                "AZURE_STORAGE_ACCOUNT_KEY"
            )
        elif auth_method == "spn":
            os.environ["AZURE_STORAGE_CLIENT_ID"] = config["env"].get(
                "AZURE_STORAGE_CLIENT_ID"
            )
            os.environ["AZURE_STORAGE_CLIENT_SECRET"] = config["env"].get(
                "AZURE_STORAGE_CLIENT_SECRET"
            )
            os.environ["AZURE_STORAGE_TENANT_ID"] = config["env"].get(
                "AZURE_STORAGE_TENANT_ID"
            )
        else:
            raise ValueError("Auth method not known")


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason="This is an integration test, add the needed environment variables in test-azure-blob.yml "
    "to run it",
)
def test_azure_blob():
    for auth_method in AUTH_METHODS:

        prepare_env(auth_method)

        blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
        blob_url = blob_path + "/" + blob_dir + "/" + blob_file

        print(f"\nBlob URL: {blob_url}")

        data_item = mlrun.run.get_dataitem(blob_url)
        data_item.put(test_string)

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
def test_list_dir():
    for auth_method in AUTH_METHODS:
        prepare_env(auth_method)
        blob_container_path = "az://" + config["env"].get("AZURE_CONTAINER")
        blob_url = blob_container_path + "/" + blob_dir + "/" + blob_file
        print(f"\nBlob URL: {blob_url}")

        mlrun.run.get_dataitem(blob_url).put(test_string)

        # Check dir list for container
        dir_list = mlrun.run.get_dataitem(blob_container_path).listdir()
        assert blob_dir + "/" + blob_file in dir_list, "File not in container dir-list"

        # Check dir list for folder in container
        dir_list = mlrun.run.get_dataitem(
            blob_container_path + "/" + blob_dir
        ).listdir()
        assert blob_file in dir_list, "File not in folder dir-list"


@pytest.mark.skipif(
    not azure_connection_configured(),
    reason="This is an integration test, add the needed environment variables in test-azure-blob.yml "
    "to run it",
)
def test_blob_upload():
    # Check upload functionality
    for auth_method in AUTH_METHODS:
        prepare_env(auth_method)

        blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
        blob_url = blob_path + "/" + blob_dir + "/" + blob_file
        print(f"\nBlob URL: {blob_url}")

        upload_data_item = mlrun.run.get_dataitem(blob_url)
        upload_data_item.upload(test_filename)

        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"
