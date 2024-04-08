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
import uuid

import dask.dataframe as dd
import fsspec
import pandas as pd
import pytest
import yaml

import mlrun
import mlrun.errors
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileS3,
    register_temporary_client_datastore_profile,
)
from mlrun.secrets import SecretsStore
from mlrun.utils import logger

here = os.path.dirname(__file__)
config_file_path = os.path.join(here, "test-aws-s3.yml")
with open(config_file_path) as yaml_file:
    config = yaml.safe_load(yaml_file)

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
    assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    env = config["env"]
    bucket_name = env.get("bucket_name")
    access_key_id = env.get("AWS_ACCESS_KEY_ID")
    _secret_access_key = env.get("AWS_SECRET_ACCESS_KEY")
    profile_name = "s3ds_profile"
    test_dir = "/test_mlrun_s3"
    run_dir = f"{test_dir}/run_{uuid.uuid4()}"
    test_file = os.path.join(assets_path, "test.txt")

    @classmethod
    def setup_class(cls):
        with open(cls.test_file) as f:
            cls.test_string = f.read()
        cls._fs = fsspec.filesystem(
            "s3", anon=False, key=cls.access_key_id, secret=cls._secret_access_key
        )

    @classmethod
    def teardown_class(cls):
        test_dir = f"{cls.bucket_name}{cls.test_dir}"
        if not cls._fs:
            return
        if cls._fs.exists(test_dir):
            cls._fs.delete(test_dir, recursive=True)
            logger.debug("test directory has been deleted.")

    def setup_method(self, method):
        store_manager.reset_secrets()
        self.profile = DatastoreProfileS3(
            name=self.profile_name,
            access_key_id=self.access_key_id,
            secret_key=self._secret_access_key,
        )
        register_temporary_client_datastore_profile(self.profile)

    def teardown_method(self, method):
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self._secret_access_key

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        mlrun.datastore.store_manager.reset_secrets()

        # We give priority to profiles, then to secrets, and finally to environment variables.
        # We want to ensure that we test these priorities in the correct order.
        if use_datastore_profile:
            os.environ["AWS_ACCESS_KEY_ID"] = "wrong_access_key"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "wrong_token"
            prefix_path = f"ds://{self.profile_name}/"
        else:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self._secret_access_key
            prefix_path = "s3://"
        self._bucket_path = f"{prefix_path}{self.bucket_name}"
        self.run_dir_url = f"{self._bucket_path}{self.run_dir}"
        object_file = f"/file_{uuid.uuid4()}.txt"
        self._object_url = f"{self.run_dir_url}{object_file}"

    def _perform_aws_s3_tests(self, secrets=None):
        #  TODO split to smaller tests, according to datastore's tests convention.
        logger.info(f"Object URL: {self._object_url}")

        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(self.test_string)
        df_url = f"{self.run_dir_url}/df_{uuid.uuid4()}.csv"
        df_data_item = mlrun.run.get_dataitem(df_url, secrets=secrets)
        df_data_item.put(test_df_string)

        response = data_item.get()
        assert response.decode() == self.test_string

        response = data_item.get(offset=20)
        assert response.decode() == self.test_string[20:]

        stat = data_item.stat()
        assert stat.size == len(self.test_string)

        dir_list = mlrun.run.get_dataitem(self.run_dir_url).listdir()

        assert self._object_url.replace(f"{self.run_dir_url}/", "") in dir_list
        assert df_url.replace(f"{self.run_dir_url}/", "") in dir_list

        blob_url = f"{self.run_dir_url}/file_{uuid.uuid4()}.blob"
        upload_data_item = mlrun.run.get_dataitem(blob_url)
        upload_data_item.upload(self.test_file)
        response = upload_data_item.get()
        assert response.decode() == self.test_string

        # Verify as_df() creates a proper DF. Note that the AWS case as_df() works through the fsspec interface, that's
        # why it's important to test it as well.
        df = df_data_item.as_df()
        assert list(df) == ["col1", "col2", "col3"]
        assert df.shape == (1, 3)

    def test_project_secrets_credentials(self):
        # This simulates running a job in a pod with project-secrets assigned to it
        for param in credential_params:
            os.environ.pop(param, None)
            os.environ[SecretsStore.k8s_env_variable_name_for_secret(param)] = config[
                "env"
            ][param]

        self._perform_aws_s3_tests()

        # cleanup
        for param in credential_params:
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param))

    def test_using_env_variables(self):
        # Use "naked" env variables, useful in client-side sdk.
        for param in credential_params:
            os.environ[param] = self.env[param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests()

        # cleanup
        for param in credential_params:
            os.environ.pop(param)

    def test_using_dataitem_secrets(
        self,
    ):
        # make sure no other auth method is configured
        for param in credential_params:
            os.environ.pop(param, None)
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        secrets = {param: self.env[param] for param in credential_params}
        self._perform_aws_s3_tests(secrets=secrets)

    @pytest.mark.skipif(
        not aws_s3_configured(extra_params=["MLRUN_AWS_ROLE_ARN"]),
        reason="Role ARN not configured",
    )
    def test_using_role_arn(
        self,
    ):
        params = credential_params.copy()
        params.append("MLRUN_AWS_ROLE_ARN")
        for param in params:
            os.environ[param] = self.env[param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests()

        # cleanup
        for param in params:
            os.environ.pop(param)

    @pytest.mark.skipif(
        not aws_s3_configured(extra_params=["AWS_PROFILE"]),
        reason="AWS profile not configured",
    )
    def test_using_profile(
        self,
    ):
        params = credential_params.copy()
        params.append("AWS_PROFILE")
        for param in params:
            os.environ[param] = self.env[param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests()

        # cleanup
        for param in params:
            os.environ.pop(param)

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
