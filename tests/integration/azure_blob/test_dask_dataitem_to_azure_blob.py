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


AUTH_METHODS_AND_REQUIRED_PARAMS = {
    "account_key": ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"],
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

    logger.info("Creating Dask Client")
    dask_cluster = os.getenv("DASK_CLUSTER")
    if dask_cluster:
        if dask_cluster.startswith("db://"):
            fn = mlrun.import_function(dask_cluster)
            client = fn._get_dask_client
        elif dask_cluster.startswith("tcp://"):
            from dask.distributed import Client

            client = Client(dask_cluster)
    else:
        from dask.distributed import Client

        client = Client()  # noqa: F841

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


def test_log_dask_to_azure(auth_method):
    verify_auth_parameters_and_configure_env(auth_method)
    artifact_path = "az://" + config["env"].get("AZURE_CONTAINER") + "/"

    A = np.random.randint(0, 100, size=(10000, 4))
    df = pd.DataFrame(data=A, columns=list("ABCD"))
    ddf = dd.from_pandas(df, npartitions=4)

    context = mlrun.get_or_create_ctx("test")
    context.log_dataset(
        key="test_data",
        df=ddf,
        artifact_path=artifact_path,
        format="parquet",
        stats=False,
    )
    print(f"testing context.artifact_path:  {artifact_path}")
    dataitem = context.get_dataitem(f"{artifact_path}test_data.parquet")
    ddf2 = dataitem.as_df(df_module=dd)
    df2 = ddf2.compute()
    pd.testing.assert_frame_equal(df, df2)


def test_log_large_dask_dataframe_to_azure(auth_method):
    # Create the environmental variables
    verify_auth_parameters_and_configure_env(auth_method)

    A = np.random.random_sample(size=(25000000, 6))
    df = pd.DataFrame(data=A, columns=list("ABCDEF"))
    ddf = dd.from_pandas(df, npartitions=10).persist()

    size = ddf.memory_usage().sum().compute()
    print(f"demo data has size:  {size // 1e6} MB")
    # Verify that the size of the dataframe is > 1GB, and so
    # will write a collection of files, instead of a single
    # file
    assert (size // 1e6) > 1100

    # Create environmental vars
    context = mlrun.get_or_create_ctx("test")

    # Define the artifact location
    target_path = "az://" + config["env"].get("AZURE_CONTAINER") + "/"

    context.log_dataset(
        key="demo_data",
        df=ddf,
        format="parquet",
        artifact_path=target_path,
        stats=True,
    )

    data_item2 = mlrun.get_dataitem(f"{target_path}demo_data.parquet")
    ddf2 = data_item2.as_df(df_module=dd)

    # Check that a collection of files is written to Azure,
    # rather than a single parquet file
    from adlfs import AzureBlobFileSystem

    fs = AzureBlobFileSystem(
        account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        tenant_id=os.getenv("AZURE_STORAGE_TENANT_ID"),
        client_id=os.getenv("AZURE_STORAGE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_STORAGE_CLIENT_SECRET"),
        sas_token=os.getenv("AZURE_STORAGE_SAS_TOKEN"),
    )
    # Verify that a directory was created, rather than a file
    path = target_path.partition("//")[2]
    path = os.path.join(path, "demo_data.parquet")
    assert fs.isdir(path) is True

    # Verify that a collection of files was written
    files = fs.ls(path)
    assert len(files) > 4

    df2 = ddf2.compute()
    df2 = df2.reset_index(drop=True)
    df = ddf.compute()
    df = df.reset_index(drop=True)
    # Verify that the returned dataframe matches the original
    pd.testing.assert_frame_equal(
        df, df2, check_index_type=False, check_less_precise=True
    )
