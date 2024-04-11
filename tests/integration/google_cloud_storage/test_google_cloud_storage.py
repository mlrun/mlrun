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
import os.path
import tempfile
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
    DatastoreProfileGCS,
    register_temporary_client_datastore_profile,
    remove_temporary_client_datastore_profile,
)
from mlrun.utils import logger

here = os.path.dirname(__file__)
config_file_path = os.path.join(here, "test-google-cloud-storage.yml")
with open(config_file_path) as yaml_file:
    config = yaml.safe_load(yaml_file)


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
    assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    bucket_name = config["env"].get("bucket_name")
    test_dir = "test_mlrun_gcs_objects"
    run_dir = f"{test_dir}/run_{uuid.uuid4()}"
    profile_name = "gcs_profile"
    credentials_path = config["env"].get("credentials_json_file")
    test_file = os.path.join(assets_path, "test.txt")

    @classmethod
    def clean_test_directory(cls):
        cls.setup_mapping = {
            "credentials_file": cls._setup_by_google_credentials_file,
            "serialized_json": cls._setup_by_serialized_json_content,
        }
        full_test_dir = f"{cls.bucket_name}/{cls.test_dir}/"
        if cls._gcs_fs.exists(full_test_dir):
            cls._gcs_fs.delete(full_test_dir, recursive=True)

    @classmethod
    def setup_class(cls):
        with open(cls.test_file) as f:
            cls.test_string = f.read()
        try:
            credentials = json.loads(cls.credentials_path)
            token = credentials
            cls.credentials = cls.credentials_path
        except json.JSONDecodeError:
            token = cls.credentials_path
            with open(cls.credentials_path) as gcs_credentials_path:
                cls.credentials = gcs_credentials_path.read()

        cls._gcs_fs = fsspec.filesystem("gcs", token=token)
        cls.clean_test_directory()

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
        object_file = f"/file_{uuid.uuid4()}.txt"
        self._object_path = f"{self.run_dir}{object_file}"
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        os.environ.pop("GCP_CREDENTIALS", None)
        remove_temporary_client_datastore_profile(self.profile_name)
        self._bucket_path = (
            f"ds://{self.profile_name}/{self.bucket_name}"
            if use_datastore_profile
            else f"gcs://{self.bucket_name}"
        )
        self.run_dir_url = f"{self._bucket_path}/{self.run_dir}"
        self._object_url = f"{self.run_dir_url}{object_file}"
        logger.info(f"Object URL: {self._object_url}")
        self.storage_options = {}

    @classmethod
    def teardown_class(cls):
        cls.clean_test_directory()

    def teardown_method(self, method):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        os.environ.pop("GCP_CREDENTIALS", None)

    def _setup_by_google_credentials_file(self, use_datastore_profile, use_secrets):
        # We give priority to profiles, then to secrets, and finally to environment variables.
        self.storage_options = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="credentials_json_file")
            if use_secrets:
                self.storage_options = {"GOOGLE_APPLICATION_CREDENTIALS": "wrong path"}
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
        else:
            if use_secrets:
                self.storage_options = {
                    "GOOGLE_APPLICATION_CREDENTIALS": self.credentials_path
                }
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

    def _setup_by_serialized_json_content(self, use_datastore_profile, use_secrets):
        self.storage_options = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="gcp_credentials")
            if use_secrets:
                self.storage_options = {"GCP_CREDENTIALS": "wrong credentials"}
            else:
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
        else:
            if use_secrets:
                self.storage_options = {"GCP_CREDENTIALS": self.credentials}
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
            else:
                os.environ["GCP_CREDENTIALS"] = self.credentials

    @pytest.mark.parametrize(
        "setup_by, use_secrets",
        [
            ("credentials_file", False),
            ("credentials_file", True),
            ("serialized_json", False),
            ("serialized_json", True),
        ],
    )
    def test_perform_google_cloud_storage_tests(
        self, use_datastore_profile, setup_by, use_secrets
    ):
        # TODO: split to smaller tests by datastore conventions
        self.setup_mapping[setup_by](self, use_datastore_profile, use_secrets)
        data_item = mlrun.run.get_dataitem(
            self._object_url, secrets=self.storage_options
        )
        data_item.put(self.test_string)

        response = data_item.get()
        assert response.decode() == self.test_string

        response = data_item.get(offset=20)
        assert response.decode() == self.test_string[20:]

        stat = data_item.stat()
        assert stat.size == len(self.test_string)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == self.test_string

        dir_list = mlrun.run.get_dataitem(
            self._bucket_path, secrets=self.storage_options
        ).listdir()
        assert self._object_path in dir_list
        listdir_dataitem_parent = mlrun.run.get_dataitem(
            os.path.dirname(self._object_url), secrets=self.storage_options
        )
        listdir_parent = listdir_dataitem_parent.listdir()
        assert os.path.basename(self._object_path) in listdir_parent

        data_item.delete()
        listdir_parent = listdir_dataitem_parent.listdir()
        assert os.path.basename(self._object_path) not in listdir_parent

        upload_data_item = mlrun.run.get_dataitem(
            self._object_url, secrets=self.storage_options
        )
        upload_data_item.upload(self.test_file)
        response = upload_data_item.get()
        assert response.decode() == self.test_string

    @pytest.mark.parametrize(
        "setup_by, use_secrets",
        [
            ("credentials_file", False),
            ("credentials_file", True),
            ("serialized_json", False),
            ("serialized_json", True),
        ],
    )
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
        use_datastore_profile,
        setup_by,
        use_secrets,
        file_format,
        pd_reader,
        dd_reader,
        reader_args,
    ):
        self.setup_mapping[setup_by](self, use_datastore_profile, use_secrets)
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = os.path.join(self.assets_path, f"test_data.{file_format}")

        source = pd_reader(local_file_path, **reader_args)
        upload_data_item = mlrun.run.get_dataitem(
            dataframe_url, secrets=self.storage_options
        )
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
    def test_as_df_directory(
        self,
        use_datastore_profile,
        file_format,
        pd_reader,
        dd_reader,
        reset_index,
    ):
        self._setup_by_google_credentials_file(
            use_datastore_profile=use_datastore_profile, use_secrets=True
        )
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = os.path.join(self.assets_path, f"test_data.{file_format}")
        df2_path = os.path.join(self.assets_path, f"additional_data.{file_format}")

        # upload
        dt1 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df1.{file_format}", secrets=self.storage_options
        )
        dt2 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df2.{file_format}", secrets=self.storage_options
        )
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        dt_dir = mlrun.run.get_dataitem(dataframes_url, secrets=self.storage_options)
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
