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
from mlrun.secrets import SecretsStore
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-aws-s3.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename, "r") as f:
    test_string = f.read()

credential_params = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


def aws_s3_configured():
    env_params = config["env"]
    needed_params = ["bucket_name", *credential_params]
    for param in needed_params:
        if not env_params.get(param):
            return False
    return True


@pytest.mark.skipif(not aws_s3_configured(), reason="AWS S3 parameters not configured")
class TestAwsS3:
    def setup_method(self, method):
        self._bucket_name = config["env"].get("bucket_name")
        self._access_key_id = config["env"].get("AWS_ACCESS_KEY_ID")
        self._secret_access_key = config["env"].get("AWS_SECRET_ACCESS_KEY")

        object_dir = "test_mlrun_s3_objects"
        object_file = f"file_{random.randint(0, 1000)}.txt"

        self._bucket_path = "s3://" + self._bucket_name
        self._object_path = object_dir + "/" + object_file
        self._object_url = self._bucket_path + "/" + self._object_path
        self._blob_url = self._object_url + ".blob"

        logger.info(f"Object URL: {self._object_url}")

    def _perform_aws_s3_tests(self, secrets=None):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
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
            os.environ[param] = config["env"][param]
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        self._perform_aws_s3_tests()

        # cleanup
        for param in credential_params:
            os.environ.pop(param)

    def test_using_dataitem_secrets(self):
        # make sure no other auth method is configured
        for param in credential_params:
            os.environ.pop(param, None)
            os.environ.pop(SecretsStore.k8s_env_variable_name_for_secret(param), None)

        secrets = {param: config["env"][param] for param in credential_params}
        self._perform_aws_s3_tests(secrets=secrets)
