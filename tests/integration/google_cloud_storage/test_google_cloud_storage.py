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
import json
import os
import tempfile
import uuid
from pathlib import Path

import fsspec
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

import mlrun
import mlrun.errors
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    DatastoreProfileGCS,
    register_temporary_client_datastore_profile,
)
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-google-cloud-storage.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename) as f:
    test_string = f.read()

credential_params = ["credentials_json_file"]


def google_cloud_storage_configured():
    env_params = config["env"]
    needed_params = ["bucket_name", *credential_params]
    for param in needed_params:
        if not env_params.get(param):
            return False
    return True


@pytest.mark.skipif(
    not google_cloud_storage_configured(),
    reason="Google cloud storage parameters not configured",
)
@pytest.mark.parametrize("use_datastore_profile", [False, True])
class TestGoogleCloudStorage:
    #  must be static, in order to get access from setup_class or teardown_class.
    @staticmethod
    def clean_test_directory(bucket_name, object_dir, gcs_fs):
        test_dir = f"{bucket_name}/{object_dir}/"
        if gcs_fs.exists(test_dir):
            gcs_fs.delete(test_dir, recursive=True)

    def setup_class(self):
        self._bucket_name = config["env"].get("bucket_name")
        self.object_dir = "test_mlrun_gcs_objects"
        self.profile_name = "gcs_profile"
        self.credentials_path = config["env"].get("credentials_json_file")

        try:
            credentials = json.loads(self.credentials_path)
            token = credentials
            self.credentials = self.credentials_path
        except json.JSONDecodeError:
            token = self.credentials_path
            with open(self.credentials_path) as gcs_credentials_path:
                self.credentials = gcs_credentials_path.read()

        self._gcs_fs = fsspec.filesystem("gcs", token=token)
        self.clean_test_directory(
            bucket_name=self._bucket_name,
            object_dir=self.object_dir,
            gcs_fs=self._gcs_fs,
        )

    def _setup_profile(self, profile_auth_by):
        if profile_auth_by:
            if profile_auth_by == "credentials_json_file":
                kwargs = {"credentials_path": self.credentials_path}
            else:
                kwargs = {"gcp_credentials": self.credentials}
            profile = DatastoreProfileGCS(name=self.profile_name, **kwargs)
            register_temporary_client_datastore_profile(profile)

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        store_manager.reset_secrets()
        object_file = f"file_{uuid.uuid4()}.txt"
        self._object_path = self.object_dir + "/" + object_file
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        os.environ.pop("GCP_CREDENTIALS", None)
        self._bucket_path = (
            f"ds://{self.profile_name}/{self._bucket_name}"
            if use_datastore_profile
            else "gcs://" + self._bucket_name
        )
        self._object_url = self._bucket_path + "/" + self._object_path
        logger.info(f"Object URL: {self._object_url}")

    def teardown_class(self):
        self.clean_test_directory(
            bucket_name=self._bucket_name,
            object_dir=self.object_dir,
            gcs_fs=self._gcs_fs,
        )

    def _perform_google_cloud_storage_tests(self, secrets={}):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(test_string)

        response = data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:], "Partial result not as expected"

        stat = data_item.stat()
        assert stat.size == len(test_string), "Stat size different than expected"

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == test_string

        dir_list = mlrun.run.get_dataitem(self._bucket_path, secrets=secrets).listdir()
        assert self._object_path in dir_list, "File not in container dir-list"
        listdir_dataitem_parent = mlrun.run.get_dataitem(
            os.path.dirname(self._object_url), secrets=secrets
        )
        listdir_parent = listdir_dataitem_parent.listdir()
        assert (
            os.path.basename(self._object_path) in listdir_parent
        ), "File not in parent dir-list"

        data_item.delete()
        listdir_parent = listdir_dataitem_parent.listdir()
        assert os.path.basename(self._object_path) not in listdir_parent

        upload_data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        upload_data_item.upload(test_filename)
        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        #  as_df tests
        upload_parquet_file_path = f"{os.path.dirname(self._object_url)}/file.parquet"
        upload_parquet_data_item = mlrun.run.get_dataitem(
            upload_parquet_file_path, secrets=secrets
        )
        test_parquet = here / "test_data.parquet"
        upload_parquet_data_item.upload(str(test_parquet))
        response = upload_parquet_data_item.as_df()
        assert pd.read_parquet(test_parquet).equals(response)

        upload_csv_file_path = f"{os.path.dirname(self._object_url)}/file.csv"
        upload_csv_data_item = mlrun.run.get_dataitem(
            upload_csv_file_path, secrets=secrets
        )
        test_csv = here / "test_data.csv"
        upload_csv_data_item.upload(str(test_csv))
        response = upload_csv_data_item.as_df()
        assert pd.read_csv(test_csv).equals(response)

        upload_json_file_path = f"{os.path.dirname(self._object_url)}/file.json"
        upload_json_data_item = mlrun.run.get_dataitem(
            upload_json_file_path, secrets=secrets
        )
        test_json = here / "test_data.json"
        upload_json_data_item.upload(str(test_json))
        response = upload_json_data_item.as_df()
        assert pd.read_json(test_json).equals(response)

    @pytest.mark.parametrize("use_secrets", (True, False))
    def test_using_google_credentials_file(self, use_datastore_profile, use_secrets):
        # We give priority to profiles, then to secrets, and finally to environment variables.
        secrets = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="credentials_json_file")
            if use_secrets:
                secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "wrong path"}
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
        else:
            if use_secrets:
                secrets = {"GOOGLE_APPLICATION_CREDENTIALS": self.credentials_path}
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        self._perform_google_cloud_storage_tests(secrets=secrets)

    @pytest.mark.parametrize("use_secrets", (True, False))
    def test_using_serialized_json_content(self, use_datastore_profile, use_secrets):
        secrets = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="gcp_credentials")
            if use_secrets:
                secrets = {"GCP_CREDENTIALS": "wrong credentials"}
            else:
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
        else:
            if use_secrets:
                secrets = {"GCP_CREDENTIALS": self.credentials}
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
            else:
                os.environ["GCP_CREDENTIALS"] = self.credentials
        self._perform_google_cloud_storage_tests(secrets=secrets)

    @pytest.mark.parametrize(
        "file_format, write_method",
        [("parquet", pd.DataFrame.to_parquet), ("csv", pd.DataFrame.to_csv)],
    )
    def test_directory(self, use_datastore_profile, file_format, write_method):
        secrets = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="credentials_json_file")
        else:
            secrets = {"GOOGLE_APPLICATION_CREDENTIALS": self.credentials_path}
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self._bucket_path}/{self.object_dir}{dataframes_dir}"
        # generate dfs
        # Define data for the first DataFrame
        data1 = {"Column1": [1, 2, 3], "Column2": ["A", "B", "C"]}
        # Define data for the second DataFrame
        data2 = {"Column1": [4, 5, 6], "Column2": ["X", "Y", "Z"]}

        # Create the DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_format}", delete=True
        ) as temp_file1, tempfile.NamedTemporaryFile(
            suffix=f".{file_format}", delete=True
        ) as temp_file2:
            # Save DataFrames as files
            write_method(df1, temp_file1.name, index=False)
            write_method(df2, temp_file2.name, index=False)
            #  upload
            dt1 = mlrun.run.get_dataitem(
                dataframes_url + f"/df1.{file_format}", secrets=secrets
            )
            dt2 = mlrun.run.get_dataitem(
                dataframes_url + f"/df2.{file_format}", secrets=secrets
            )
            dt1.upload(src_path=temp_file1.name)
            dt2.upload(src_path=temp_file2.name)
            dt_dir = mlrun.run.get_dataitem(dataframes_url, secrets=secrets)
            tested_df = dt_dir.as_df(format=f"{file_format}")
            expected_df = pd.concat([df1, df2], ignore_index=True)
            assert_frame_equal(
                tested_df.reset_index(drop=True), expected_df, check_like=True
            )
