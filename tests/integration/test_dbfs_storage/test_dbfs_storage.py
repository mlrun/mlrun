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
from pathlib import Path

import pytest
import yaml

import mlrun
import mlrun.errors
from mlrun.utils import logger

here = Path(__file__).absolute().parent
config_file_path = here / "test-dbfs-storage.yml"
with config_file_path.open() as fp:
    config = yaml.safe_load(fp)

test_filename = here / "test.txt"
with open(test_filename, "r") as f:
    test_string = f.read()

MUST_HAVE_VARIABLES = ["DATABRICKS_TOKEN", "DATABRICKS_WORKSPACE"]


# def configure_dbfs_storage():
#     env_params = config["env"]
#     for key, env_param in env_params.items():
#         os.environ[key] = env_params


def is_dbfs_configured():
    env_params = config["env"]
    for necessary_variable in MUST_HAVE_VARIABLES:
        if env_params.get(necessary_variable, None) is None:
            return False
    return True


@pytest.mark.skipif(
    not is_dbfs_configured(),
    reason="DBFS storage parameters not configured",
)
class TestDBFSStorage:
    def setup_method(self):
        self._databricks_workspace = config["env"].get("DATABRICKS_WORKSPACE")
        self._object_dir = "/test_mlrun_dbfs_objects"
        self._object_file = f"file_{str(uuid.uuid4())}.txt"
        self._object_path = self._object_dir + "/" + self._object_file
        self._dbfs_url = "dbfs://" + self._databricks_workspace
        self._object_url = self._dbfs_url + self._object_path

        logger.info(f"Object URL: {self._object_url}")

    def _perform_dbfs_tests(self, secrets):
        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
        data_item.put(test_string)
        response = data_item.get()
        assert response.decode() == test_string, "Result differs from original test"
        #
        response = data_item.get(offset=20)
        assert response.decode() == test_string[20:], "Partial result not as expected"
        #
        stat = data_item.stat()
        assert stat.size == len(test_string), "Stat size different than expected"

        dir_dataitem = mlrun.run.get_dataitem(self._dbfs_url + self._object_dir)
        dir_list = dir_dataitem.listdir()
        assert self._object_file in dir_list, "File not in container dir-list"

        upload_file_path = self._object_dir + "/" + f"file_{str(uuid.uuid4())}.txt"
        upload_data_item = mlrun.run.get_dataitem(self._dbfs_url + upload_file_path)
        upload_data_item.upload(str(test_filename))
        response = upload_data_item.get()
        assert response.decode() == test_string, "Result differs from original test"

    def test_secrets_as_input(self):
        secrets = {}
        token = config["env"].get("DATABRICKS_TOKEN", None)
        secrets["DATABRICKS_TOKEN"] = token
        self._perform_dbfs_tests(secrets=secrets)

    def test_using_dbfs_env_variable(self):
        env_params = config["env"]
        for key, env_param in env_params.items():
            os.environ[key] = env_param
        self._perform_dbfs_tests(secrets={})
