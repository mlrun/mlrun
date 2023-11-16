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
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

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
with open(test_filename, "r") as f:
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
@pytest.mark.parametrize(
    "use_datastore_profile_by", [None, "credentials_json_file", "gcp_credentials"]
)
class TestGoogleCloudStorage:
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile_by):
        store_manager.reset_secrets()
        self._bucket_name = config["env"].get("bucket_name")
        object_dir = "test_mlrun_gcs_objects"
        object_file = f"file_{random.randint(0, 1000)}.txt"
        self._object_path = object_dir + "/" + object_file
        self.profile_name = "gcs_profile"
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        os.environ.pop("GCP_CREDENTIALS", None)
        if use_datastore_profile_by:
            if use_datastore_profile_by == "credentials_json_file":
                kwargs = {
                    "credentials_path": config["env"].get("credentials_json_file")
                }
            else:
                with open(config["env"].get("credentials_json_file"), "r") as f:
                    credentials = f.read()
                kwargs = {"gcp_credentials": credentials}
            profile = DatastoreProfileGCS(name=self.profile_name, **kwargs)
            register_temporary_client_datastore_profile(profile)

        self._bucket_path = (
            f"ds://{self.profile_name}/{self._bucket_name}"
            if use_datastore_profile_by
            else "gcs://" + self._bucket_name
        )
        self._object_url = self._bucket_path + "/" + self._object_path
        self._blob_url = self._object_url + ".blob"
        logger.info(f"Object URL: {self._object_url}")

    def _perform_google_cloud_storage_tests(self, secrets={}):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(test_string)

        response = data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:], "Partial result not as expected"

        stat = data_item.stat()
        assert stat.size == len(test_string), "Stat size different than expected"

        dir_list = mlrun.run.get_dataitem(self._bucket_path, secrets=secrets).listdir()
        assert self._object_path in dir_list, "File not in container dir-list"
        listdir_parent = mlrun.run.get_dataitem(
            os.path.dirname(self._object_url), secrets=secrets
        ).listdir()
        assert (
            os.path.basename(self._object_path) in listdir_parent
        ), "File not in parent dir-list"

        upload_data_item = mlrun.run.get_dataitem(self._blob_url, secrets=secrets)
        upload_data_item.upload(test_filename)
        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"
        upload_parquet_file_path = f"{os.path.dirname(self._blob_url)}/file.parquet"
        upload_parquet_data_item = mlrun.run.get_dataitem(
            upload_parquet_file_path, secrets=secrets
        )
        test_parquet = here / "test_data.parquet"
        upload_parquet_data_item.upload(str(test_parquet))
        response = upload_parquet_data_item.as_df()
        assert pd.read_parquet(test_parquet).equals(response)
        upload_csv_file_path = f"{os.path.dirname(self._blob_url)}/file.csv"
        upload_csv_data_item = mlrun.run.get_dataitem(
            upload_csv_file_path, secrets=secrets
        )
        test_csv = here / "test_data.csv"
        upload_csv_data_item.upload(str(test_csv))
        response = upload_csv_data_item.as_df()
        assert pd.read_csv(test_csv).equals(response)

    @pytest.mark.parametrize("use_secrets", (True, False))
    def test_using_google_credentials_file(self, use_datastore_profile_by, use_secrets):
        # We give priority to profiles, then to secrets, and finally to environment variables.
        secrets = {}
        credentials_json_file = config["env"].get("credentials_json_file")
        if use_datastore_profile_by:
            if use_secrets:
                secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "wrong path"}
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
        else:
            if use_secrets:
                secrets = {"GOOGLE_APPLICATION_CREDENTIALS": credentials_json_file}
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wrong path"
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json_file
        self._perform_google_cloud_storage_tests(secrets=secrets)

    @pytest.mark.parametrize("use_secrets", (True, False))
    def test_using_serialized_json_content(self, use_datastore_profile_by, use_secrets):
        secrets = {}
        with open(config["env"].get("credentials_json_file"), "r") as f:
            credentials = f.read()
        if use_datastore_profile_by:
            if use_secrets:
                secrets = {"GCP_CREDENTIALS": "wrong credentials"}
            else:
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
        else:
            if use_secrets:
                secrets = {"GCP_CREDENTIALS": credentials}
                os.environ["GCP_CREDENTIALS"] = "wrong credentials"
            else:
                os.environ["GCP_CREDENTIALS"] = credentials
        self._perform_google_cloud_storage_tests(secrets=secrets)
