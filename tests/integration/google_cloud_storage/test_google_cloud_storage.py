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
import random
from pathlib import Path

import pytest
import yaml

import mlrun
import mlrun.errors
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
class TestGoogleCloudStorage:
    def setup_method(self, method):
        self._bucket_name = config["env"].get("bucket_name")

        object_dir = "test_mlrun_gcs_objects"
        object_file = f"file_{random.randint(0, 1000)}.txt"

        self._bucket_path = "gcs://" + self._bucket_name
        self._object_path = object_dir + "/" + object_file
        self._object_url = self._bucket_path + "/" + self._object_path
        self._blob_url = self._object_url + ".blob"

        logger.info(f"Object URL: {self._object_url}")

    def _perform_google_cloud_storage_tests(self):
        data_item = mlrun.run.get_dataitem(self._object_url)
        data_item.put(test_string)

        response = data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:], "Partial result not as expected"

        stat = data_item.stat()
        assert stat.size == len(test_string), "Stat size different than expected"

        dir_list = mlrun.run.get_dataitem(self._bucket_path).listdir()
        assert self._object_path in dir_list, "File not in container dir-list"

        upload_data_item = mlrun.run.get_dataitem(self._blob_url)
        upload_data_item.upload(test_filename)
        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

    def test_using_google_env_variable(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["env"].get(
            "credentials_json_file"
        )
        self._perform_google_cloud_storage_tests()

    def test_using_serialized_json_content(self):
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        with open(config["env"].get("credentials_json_file"), "r") as f:
            credentials = f.read()
        os.environ["GCP_CREDENTIALS"] = credentials
        self._perform_google_cloud_storage_tests()
