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
    @classmethod
    def clean_test_directory(cls):
        full_test_dir = f"{cls._bucket_name}/{cls.test_dir}/"
        if cls._gcs_fs.exists(full_test_dir):
            cls._gcs_fs.delete(full_test_dir, recursive=True)

    @classmethod
    def setup_class(cls):
        cls.assets_path = join(dirname(dirname(abspath(__file__))), "assets")
        cls._bucket_name = config["env"].get("bucket_name")
        cls.test_dir = "test_mlrun_gcs_objects"
        cls.run_dir = cls.test_dir + f"/run_{uuid.uuid4()}"
        cls.profile_name = "gcs_profile"
        cls.credentials_path = config["env"].get("credentials_json_file")
        cls.setup_mapping = {
            "credentials_file": cls._setup_by_google_credentials_file,
            "serialized_json": cls._setup_by_serialized_json_content,
        }

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
        object_file = f"file_{uuid.uuid4()}.txt"
        self._object_path = self.run_dir + "/" + object_file
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        os.environ.pop("GCP_CREDENTIALS", None)
        self._bucket_path = (
            f"ds://{self.profile_name}/{self._bucket_name}"
            if use_datastore_profile
            else "gcs://" + self._bucket_name
        )
        self.run_dir_url = f"{self._bucket_path}/{self.run_dir}"
        self._object_url = self.run_dir_url + "/" + object_file
        logger.info(f"Object URL: {self._object_url}")

    @classmethod
    def teardown_class(cls):
        cls.clean_test_directory()

    def _setup_by_google_credentials_file(
        self, use_datastore_profile, use_secrets
    ) -> dict:
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
        return secrets

    def _setup_by_serialized_json_content(self, use_datastore_profile, use_secrets):
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
        return secrets

    def _setup_df_dir(self, use_datastore_profile, file_format, reader):
        secrets = {}
        if use_datastore_profile:
            self._setup_profile(profile_auth_by="credentials_json_file")
        else:
            secrets = {"GOOGLE_APPLICATION_CREDENTIALS": self.credentials_path}
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = join(self.assets_path, f"test_data.{file_format}")
        df2_path = join(self.assets_path, f"additional_data.{file_format}")

        #  upload
        dt1 = mlrun.run.get_dataitem(
            dataframes_url + f"/df1.{file_format}", secrets=secrets
        )
        dt2 = mlrun.run.get_dataitem(
            dataframes_url + f"/df2.{file_format}", secrets=secrets
        )
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        return (
            mlrun.run.get_dataitem(dataframes_url),
            reader(df1_path),
            reader(df2_path),
        )

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
        #  TODO split to smaller tests by datastore conventions.
        secrets = self.setup_mapping[setup_by](self, use_datastore_profile, use_secrets)
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
        setup_by,
        use_secrets,
        file_format,
        reader,
        reader_args,
    ):
        secrets = self.setup_mapping[setup_by](self, use_datastore_profile, use_secrets)
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)

        upload_data_item = mlrun.run.get_dataitem(dataframe_url, secrets=secrets)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(**reader_args)
        pd.testing.assert_frame_equal(source, response)

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
        setup_by,
        use_secrets,
        file_format,
        reader,
        reader_args,
    ):
        secrets = self.setup_mapping[setup_by](self, use_datastore_profile, use_secrets)
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = join(self.assets_path, f"test_data.{file_format}")
        source = reader(local_file_path, **reader_args)

        upload_data_item = mlrun.run.get_dataitem(dataframe_url, secrets=secrets)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(**reader_args, df_module=dd)
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
        # We have already checked the functionality of different setups with single file as_df tests,
        # so we do not need to do so here too.

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
