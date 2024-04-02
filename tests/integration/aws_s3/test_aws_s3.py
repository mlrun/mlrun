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
import uuid
from os.path import abspath, dirname, join
from pathlib import Path

import dask.dataframe as dd
import fsspec
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

import mlrun
import mlrun.errors
from mlrun.datastore import store_manager
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
    def _make_target_names(self, prefix, object_file, csv_file):
        bucket_path = prefix + self._bucket_name
        object_path = f"{self.run_dir}/{object_file}"
        df_path = f"{self.run_dir}/{csv_file}"
        object_url = f"{bucket_path}/{object_path}"
        res = {
            "bucket_path": bucket_path,
            "object_path": object_path,
            "df_path": df_path,
            "object_url": object_url,
            "df_url": f"{bucket_path}/{df_path}",
            "blob_url": f"{object_url}.blob",
        }
        return res

    @classmethod
    def setup_class(cls):
        cls.assets_path = join(dirname(dirname(abspath(__file__))), "assets")
        cls._bucket_name = config["env"].get("bucket_name")
        cls._access_key_id = config["env"].get("AWS_ACCESS_KEY_ID")
        cls._secret_access_key = config["env"].get("AWS_SECRET_ACCESS_KEY")
        cls.profile_name = "s3ds_profile"
        cls.test_dir = "/test_mlrun_s3"
        cls.run_dir = cls.test_dir + f"/run_{uuid.uuid4()}"
        cls.test_file = join(cls.assets_path, "test.txt")
        with open(cls.test_file) as f:
            cls.test_string = f.read()
        cls._fs = fsspec.filesystem(
            "s3", anon=False, key=cls._access_key_id, secret=cls._secret_access_key
        )

    @classmethod
    def teardown_class(cls):
        test_dir = f"{cls._bucket_name}{cls.test_dir}"
        if not cls._fs:
            return
        if cls._fs.exists(test_dir):
            cls._fs.delete(test_dir, recursive=True)
            logger.debug("test directory has been cleaned.")

    def setup_method(self, method):
        object_file = f"file_{uuid.uuid4()}.txt"
        csv_file = f"file_{uuid.uuid4()}.csv"

        self.s3 = {}
        self.s3["s3"] = self._make_target_names("s3://", object_file, csv_file)
        self.s3["ds"] = self._make_target_names(
            f"ds://{self.profile_name}/", object_file, csv_file
        )
        store_manager.reset_secrets()
        self.profile = DatastoreProfileS3(
            name=self.profile_name,
            access_key_id=self._access_key_id,
            secret_key=self._secret_access_key,
        )
        register_temporary_client_datastore_profile(self.profile)

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        if use_datastore_profile:
            os.environ["AWS_ACCESS_KEY_ID"] = "wrong_access_key"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "wrong_token"
        else:
            os.environ["AWS_ACCESS_KEY_ID"] = self._access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self._secret_access_key

    def _setup_df_dir(self, use_datastore_profile, file_format, reader):
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        directory = f"/{file_format}s_{uuid.uuid4()}"
        s3_directory_url = param["bucket_path"] + self.run_dir + directory
        df1_path = join(self.assets_path, f"test_data.{file_format}")
        df2_path = join(self.assets_path, f"additional_data.{file_format}")

        #  upload
        dt1 = mlrun.run.get_dataitem(s3_directory_url + f"/df1.{file_format}")
        dt2 = mlrun.run.get_dataitem(s3_directory_url + f"/df2.{file_format}")
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        return (
            mlrun.run.get_dataitem(s3_directory_url),
            reader(df1_path),
            reader(df2_path),
        )

    def _perform_aws_s3_tests(self, use_datastore_profile, secrets=None):
        #  TODO split to smaller tests, like datastore's tests convention.
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        logger.info(f'Object URL: {param["object_url"]}')

        data_item = mlrun.run.get_dataitem(param["object_url"], secrets=secrets)
        data_item.put(self.test_string)
        df_data_item = mlrun.run.get_dataitem(param["df_url"], secrets=secrets)
        df_data_item.put(test_df_string)

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

        dir_list = mlrun.run.get_dataitem(param["bucket_path"]).listdir()
        assert param["object_path"] in dir_list, "File not in container dir-list"
        assert param["df_path"] in dir_list, "CSV file not in container dir-list"

        upload_data_item = mlrun.run.get_dataitem(param["blob_url"])
        upload_data_item.upload(self.test_file)
        response = upload_data_item.get()
        assert (
            response.decode() == self.test_string
        ), "Result differs from original test"

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
        "file_format, reader, reader_args",
        [
            ("parquet", pd.read_parquet, {}),
            ("csv", pd.read_csv, {}),
            ("json", pd.read_json, {"orient": "values"}),
        ],
    )
    def test_as_df(
        self,
        use_datastore_profile,
        file_format: str,
        reader: callable,
        reader_args: dict,
    ):
        # A more advanced test of the as_df function that includes Parquet, CSV, and JSON files.
        param = self.s3["ds"] if use_datastore_profile else self.s3["s3"]
        for p in credential_params:
            os.environ[p] = config["env"][p]
        filename = f"/df_{uuid.uuid4()}.{file_format}"
        file_url = param["bucket_path"] + self.run_dir + filename

        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)

        upload_data_item = mlrun.run.get_dataitem(file_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(**reader_args)
        pd.testing.assert_frame_equal(source, response)

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
        file_url = param["bucket_path"] + self.run_dir + filename

        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)

        upload_data_item = mlrun.run.get_dataitem(file_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(df_module=dd, **reader_args)
        dd.assert_eq(source, response)

    @pytest.mark.parametrize(
        "file_format, reader, reset_index",
        [
            ("parquet", pd.read_parquet, False),
            ("csv", pd.read_csv, True),
        ],
    )
    def test_as_df_directory(
        self, use_datastore_profile, file_format, reader, reset_index
    ):
        for p in credential_params:
            os.environ[p] = config["env"][p]
        dt_dir, df1, df2 = self._setup_df_dir(
            use_datastore_profile=use_datastore_profile,
            file_format=file_format,
            reader=reader,
        )
        tested_df = dt_dir.as_df(format=file_format)
        if reset_index:
            tested_df = tested_df.sort_values("ID").reset_index(drop=True)
        expected_df = pd.concat([df1, df2], ignore_index=True)
        assert_frame_equal(tested_df, expected_df)

    @pytest.mark.parametrize(
        "file_format, reader, reset_index",
        [
            ("parquet", dd.read_parquet, False),
            ("csv", dd.read_csv, True),
        ],
    )
    def test_as_df_directory_dd(
        self, use_datastore_profile, file_format, reader, reset_index
    ):
        for p in credential_params:
            os.environ[p] = config["env"][p]
        dt_dir, df1, df2 = self._setup_df_dir(
            use_datastore_profile=use_datastore_profile,
            file_format=file_format,
            reader=reader,
        )
        tested_df = dt_dir.as_df(format=file_format, df_module=dd)
        if reset_index:
            tested_df = tested_df.sort_values("ID").reset_index(drop=True)
        expected_df = dd.concat([df1, df2], axis=0)
        dd.assert_eq(tested_df, expected_df)
