import os
import random
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
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
    "conn_str": ["AZURE_STORAGE_CONNECTION_STRING"],
    "sas_token": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_SAS_TOKEN"],
    "account_key": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"],
    "spn": [
        "AZURE_STORAGE_ACCOUNT_NAME",
        "AZURE_STORAGE_CLIENT_ID",
        "AZURE_STORAGE_CLIENT_SECRET",
        "AZURE_STORAGE_TENANT_ID",
    ],
}


def verify_auth_parameters_and_configure_env(auth_method):
    if not config["env"].get("AZURE_CONTAINER"):
        return False

    for k, env_vars in AUTH_METHODS_AND_REQUIRED_PARAMS.items():
        for env_var in env_vars:
            os.environ.pop(env_var, None)

    test_params = AUTH_METHODS_AND_REQUIRED_PARAMS.get(auth_method)
    if not test_params:
        return False

    for env_var in test_params:
        env_value = config["env"].get(env_var)
        if not env_value:
            return False
        os.environ[env_var] = env_value

    logger.info(f"Testing auth method {auth_method}")
    return True


# Apply parametrization to all tests in this file. Skip test if auth method is not configured.
pytestmark = pytest.mark.parametrize(
    "auth_method",
    [
        pytest.param(
            auth_method,
            marks=pytest.mark.skipif(
                not verify_auth_parameters_and_configure_env(auth_method),
                reason=f"Auth method {auth_method} not configured.",
            ),
        )
        for auth_method in AUTH_METHODS_AND_REQUIRED_PARAMS
    ],
)


def test_azure_blob(auth_method):
    verify_auth_parameters_and_configure_env(auth_method)
    blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
    blob_url = blob_path + "/" + blob_dir + "/" + blob_file

    print(f"\nBlob URL: {blob_url}")

    data_item = mlrun.run.get_dataitem(blob_url)
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


# def test_list_dir(auth_method):
#     verify_auth_parameters_and_configure_env(auth_method)
#     blob_container_path = "az://" + config["env"].get("AZURE_CONTAINER")
#     blob_url = blob_container_path + "/" + blob_dir + "/" + blob_file
#     print(f"\nBlob URL: {blob_url}")

#     mlrun.run.get_dataitem(blob_url).put(test_string)

#     # Check dir list for container
#     dir_list = mlrun.run.get_dataitem(blob_container_path).listdir()
#     assert blob_dir + "/" + blob_file in dir_list, "File not in container dir-list"

#     # Check dir list for folder in container
#     dir_list = mlrun.run.get_dataitem(blob_container_path + "/" + blob_dir).listdir()
#     assert blob_file in dir_list, "File not in folder dir-list"


# def test_blob_upload(auth_method):
#     verify_auth_parameters_and_configure_env(auth_method)
#     blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
#     blob_url = blob_path + "/" + blob_dir + "/" + blob_file
#     print(f"\nBlob URL: {blob_url}")

#     upload_data_item = mlrun.run.get_dataitem(blob_url)
#     upload_data_item.upload(test_filename)

#     response = upload_data_item.get()
#     assert response.decode() == test_string, "Result differs from original test"


# def test_log_dask_to_azure(auth_method):
#     verify_auth_parameters_and_configure_env(auth_method)
#     blob_path = "az://" + config["env"].get("AZURE_CONTAINER")

#     A = np.random.randint(0, 100, size=(10000, 4))
#     df = pd.DataFrame(data=A, columns=list("ABCD"))
#     ddf = dd.from_pandas(df, npartitions=4)

#     context = mlrun.get_or_create_ctx("test")
#     context.log_dataset(
#         key="test_data",
#         df=ddf,
#         artifact_path=f"az://{blob_path}/",
#         format="parquet",
#         stats=False,
#     )
#     dataitem = context.get_dataitem(f"{context.artifact_path}test_data.parquet")
#     ddf2 = dataitem.as_df(df_module=dd)
#     df2 = ddf2.compute()
#     pd.testing.assert_frame_equal(df, df2)


# def test_log_large_dask_dataframe_to_azure(auth_method):
    # # Load a parquet file from Azure Open Datasets
    # os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "azureopendatastorage"
    # print("Fetching demo_data from Azure Open Datasets")
    # data_item = mlrun.datastore.store_manager.object(
    #     "az://tutorials/noaa_isd_weather/demo_data.parquet"
    # )
    # ddf = data_item.as_df(df_module=dd)
    # print(f"open_data has {ddf.npartitions} partitions")
    # mem_size = ddf.memory_usage().sum().compute()
    # print(f"demo data has size:  {mem_size}")

    # # Create environmental vars
    # verify_auth_parameters_and_configure_env(auth_method)
    # context = mlrun.get_or_create_ctx("test")

    # # Define the artifact location
    # blob_path = "az://" + config["env"].get("AZURE_CONTAINER")
    # target_path=f"az://{blob_path}/"

    # mlrun.log_dataset(
    #     key="demo_data",
    #     df = ddf,
    #     format="parquet",
    #     artifact_path=target_path,
    # )
    
    # data_item2 = mlrun.get_dataitem(f"{target_path}demo_data.parquet")
    # ddf2 = data_item2.get(df_module=dd)
    # mem_size2 = ddf2.memory_usage().sum().compute()
    # print(f"size of mlrun dataitem is:  {mem_size2}")
    # assert mem_size == mem_size2
    
    # # Check the # of partitions in ddf2 vs the # of files written
    # from adlfs import AzureBlobFileSystem
    # fs = AzureBlobFileSystem(account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
    #                          account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
    #                          connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    #                          tenant_id=os.getenv("AZURE_STORAGE_TENANT_ID"),
    #                          client_id=os.getenv("AZURE_STORAGE_CLIENT_ID"),
    #                          client_secret=os.getenv("AZURE_STORAGE_CLIENT_SECRET"),
    #                          sas_token=os.getenv("AZURE_STORAGE_SAS_TOKEN")
    #                          )
    # files = fs.ls(f"{target_path}demo_data.parquet")
    # assert ddf2.npartitions == len(files)
    
