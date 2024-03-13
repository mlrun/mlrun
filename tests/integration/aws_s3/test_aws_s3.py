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
from os.path import dirname, abspath, join
import random
import tempfile
import uuid
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

import mlrun
import mlrun.errors
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
from mlrun.secrets import SecretsStore
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-aws-s3.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename) as f:
    test_string = f.read()

# Used to test dataframe functionality (will be saved as csv)
test_df_string = "col1,col2,col3\n1,2,3"

credential_params = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def aws_s3_configured(extra_params=None):
    extra_params = extra_params or []
    env_params = config["env"]
    needed_params = ["bucket_name", *credential_params, *extra_params]
    for param in needed_params:
        if not env_params.get(param):
            return False
    return True


@pytest.mark.skipif(not aws_s3_configured(), reason="AWS S3 parameters not configured")
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestAwsS3:
    def _make_target_names(
        self, prefix, bucket_name, object_dir, object_file, csv_file
    ):
        bucket_path = prefix + bucket_name
        object_path = f"{object_dir}/{object_file}"
        df_path = f"{object_dir}/{csv_file}"
        object_url = f"{bucket_path}/{object_path}"
        res = {
            "bucket_path": bucket_path,
            "object_path": object_path,
            "df_path": df_path,
            "object_url": object_url,
            "df_url": f"{bucket_path}/{df_path}",
            "blob_url": f"{object_url}.blob",
            "parquet_url": f"{object_url}.parquet",
        }
        return res

    def setup_class(self):
        self.assets_path = join(dirname(dirname(abspath(__file__))), "assets")
        # TODO move setup to here from setup_method

    def setup_method(self, method):
        self._bucket_name = config["env"].get("bucket_name")
        self._access_key_id = config["env"].get("AWS_ACCESS_KEY_ID")
        self._secret_access_key = config["env"].get("AWS_SECRET_ACCESS_KEY")

        object_dir = "test_mlrun_s3_objects"
        object_file = f"file_{random.randint(0, 1000)}.txt"
        csv_file = f"file_{random.randint(0,1000)}.csv"

        self.s3 = {}
        self.s3["s3"] = self._make_target_names(
            "s3://", self._bucket_name, object_dir, object_file, csv_file
        )
        self.s3["ds"] = self._make_target_names(
            "ds://s3ds_profile/", self._bucket_name, object_dir, object_file, csv_file
        )
        profile = DatastoreProfileS3(
            name="s3ds_profile",
            access_key_id=self._access_key_id,
            secret_key=self._secret_access_key,
        )
        register_temporary_client_datastore_profile(profile)

    def _perform_aws_s3_tests(self, use_datastore_profile, secrets=None):
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        logger.info(f'Object URL: {param["object_url"]}')

        data_item = mlrun.run.get_dataitem(param["object_url"], secrets=secrets)
        data_item.put(test_string)
        df_data_item = mlrun.run.get_dataitem(param["df_url"], secrets=secrets)
        df_data_item.put(test_df_string)

        response = data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:], "Partial result not as expected"

        stat = data_item.stat()
        assert stat.size == len(test_string), "Stat size different than expected"

        dir_list = mlrun.run.get_dataitem(param["bucket_path"]).listdir()
        assert param["object_path"] in dir_list, "File not in container dir-list"
        assert param["df_path"] in dir_list, "CSV file not in container dir-list"

        upload_data_item = mlrun.run.get_dataitem(param["blob_url"])
        upload_data_item.upload(test_filename)
        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        # Verify as_df() creates a proper DF. Note that the AWS case as_df() works through the fsspec interface, that's
        # why it's important to test it as well.
        df = df_data_item.as_df()
        assert list(df) == ["col1", "col2", "col3"]
        assert df.shape == (1, 3)

    def test_project_secrets_credentials(self, use_datastore_profile):
        # This simulates running a job in a pod with project-secrets assigned to it
        for param in credential_params:
            os.environ.pop(param, None)
            os.environ[SecretsStore.k8s_env_variable_name_for_secret(param)] = config[
                "env"
            ][param]

        self._perform_aws_s3_tests(use_datastore_profile)

        # cleanup
        for param in credential_params:
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param))

    def test_using_env_variables(self, use_datastore_profile):
        # Use "naked" env variables, useful in client-side sdk.
        for param in credential_params:
            os.environ[param] = config["env"][param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests(use_datastore_profile)

        # cleanup
        for param in credential_params:
            os.environ.pop(param)

    def test_using_dataitem_secrets(self, use_datastore_profile):
        # make sure no other auth method is configured
        for param in credential_params:
            os.environ.pop(param, None)
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        secrets = {param: config["env"][param] for param in credential_params}
        self._perform_aws_s3_tests(use_datastore_profile, secrets=secrets)

    @pytest.mark.skipif(
        not aws_s3_configured(extra_params=["MLRUN_AWS_ROLE_ARN"]),
        reason="Role ARN not configured",
    )
    def test_using_role_arn(self, use_datastore_profile):
        params = credential_params.copy()
        params.append("MLRUN_AWS_ROLE_ARN")
        for param in params:
            os.environ[param] = config["env"][param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests(use_datastore_profile)

        # cleanup
        for param in params:
            os.environ.pop(param)

    @pytest.mark.skipif(
        not aws_s3_configured(extra_params=["AWS_PROFILE"]),
        reason="AWS profile not configured",
    )
    def test_using_profile(self, use_datastore_profile):
        params = credential_params.copy()
        params.append("AWS_PROFILE")
        for param in params:
            os.environ[param] = config["env"][param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests(use_datastore_profile)

        # cleanup
        for param in params:
            os.environ.pop(param)

    @pytest.mark.parametrize(
        "file_format, reader, reset_index",
        [
            ("parquet", pd.read_parquet, False),
            ("csv", pd.read_csv, True),
        ],
    )
    def test_as_df_directory(self, use_datastore_profile, file_format, reader, reset_index):
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        for p in credential_params:
            os.environ[p] = config["env"][p]
        directory = f"/{file_format}s_{uuid.uuid4()}"
        s3_directory_url = param["bucket_path"] + directory
        # Create the DataFrames
        df1_path = join(self.assets_path, f"test_data.{file_format}")
        df2_path = join(self.assets_path, f"additional_data.{file_format}")
        df1 = reader(df1_path)
        df2 = reader(df2_path)

        #  upload
        dt1 = mlrun.run.get_dataitem(s3_directory_url + f"/df1.{file_format}")
        dt2 = mlrun.run.get_dataitem(s3_directory_url + f"/df2.{file_format}")
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        dt1.as_df()
        dt2.as_df()
        dt_dir = mlrun.run.get_dataitem(s3_directory_url)
        tested_df = dt_dir.as_df(format=file_format)
        if reset_index:
            tested_df = tested_df.reset_index(drop=True)
        expected_df = pd.concat([df1, df2], ignore_index=True)
        assert_frame_equal(tested_df, expected_df)

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
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        for p in credential_params:
            os.environ[p] = config["env"][p]
        filename = f"/df_{uuid.uuid4()}.{file_format}"
        file_url = param["bucket_path"] + filename

        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)

        upload_data_item = mlrun.run.get_dataitem(file_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd, **reader_args)
        assert dd.assert_eq(source, response)

