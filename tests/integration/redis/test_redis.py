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
import os
import tempfile
import uuid

import pytest
from pydantic.error_wrappers import ValidationError
from redis.exceptions import AuthenticationError

import mlrun.datastore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileRedis,
    register_temporary_client_datastore_profile,
)
import yaml

here = os.path.dirname(__file__)
config_file_path = os.path.join(here, "test-redis.yml")

config = {}
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

#redis_endpoint = ("redis://", "redis://localhost:6379")


@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestRedisDataStore:
    profile_name = "redis_profile"
    @classmethod
    def setup_class(cls):
        cls.redis_endpoint = os.environ["REDIS_ENDPOINT"]

    def setup_before_test(self, use_datastore_profile, redis_endpoint):
        file = f"file_{uuid.uuid4()}.txt"
        if use_datastore_profile:
            profile = DatastoreProfileRedis(
                name=self.profile_name, endpoint_url=redis_endpoint
            )
            register_temporary_client_datastore_profile(profile)
            self.endpoint = f"ds://{self.profile_name}"
        else:
            self.endpoint = redis_endpoint
        self.redis_path = f"{self.endpoint}/{file}"

    @pytest.mark.parametrize("redis_endpoint", redis_endpoint)
    def test_redis_put_get_object(self, use_datastore_profile, redis_endpoint):
        self.setup_before_test(use_datastore_profile, redis_endpoint)
        redis_path = redis_endpoint + "/redis_object"
        data_item = mlrun.datastore.store_manager.object(redis_path)

        data_item.delete()

        str_arr = ["hi ", "there, ", "stranger"]
        for i, string in enumerate(str_arr):
            data_item.put(string, append=True if i != 0 else False)

        # no "size", "offset": return the entire object
        object_value = data_item.get()
        assert object_value == "".join(str_arr)
        # no "size": returns from "offset" to end of object
        object_value = data_item.get(offset=len(str_arr[0]))
        assert object_value == "".join(str_arr[1:])
        # "size"=0 is forbidden
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            object_value = data_item.get(offset=1, size=0)
        # "size">0: return the first "size" bytes
        object_value = data_item.get(size=len(str_arr[0]))
        assert object_value == str_arr[0]
        # "size">0 "offset">0: return "size" bytes starting from byte "offset"
        object_value = data_item.get(offset=len(str_arr[0]), size=len(str_arr[1]))
        assert object_value == str_arr[1]

        data_item.delete()

    @pytest.mark.parametrize("redis_endpoint", redis_endpoint)
    def test_redis_upload_download_object(self, redis_endpoint, use_datastore_profile):
        self.setup_before_test(use_datastore_profile, redis_endpoint)
        # prepare file for upload
        expected = "abcde" * 100
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as temp_file:
            with open(temp_file.name, "w") as f:
                f.write(expected)
            data_item = mlrun.datastore.store_manager.object(self.redis_path)
            data_item.delete()

            data_item.upload(temp_file.name)

        data_item.delete()

        assert expected == expected

    @pytest.mark.parametrize("redis_endpoint", redis_endpoint)
    def test_redis_listdir(self, redis_endpoint, use_datastore_profile):
        self.setup_before_test(use_datastore_profile, redis_endpoint)
        list_dir = self.endpoint + "/dir-0/dir-1"

        redis_path = self.endpoint
        dir_path = redis_path
        expected = []

        for depth in range(5):
            dir_path = dir_path + f"/dir-{depth}"
            obj_path = dir_path + f"/obj-{depth}"
            data_item = mlrun.datastore.store_manager.object(obj_path)
            data_item.delete()
            data_item.put("abcde")
            # list_dir skips the first object
            if depth > 0:
                expected.append(obj_path[len(redis_path) :])

        dir_item = mlrun.datastore.store_manager.object(list_dir)
        actual = dir_item.listdir()
        assert set(expected) == set(
            actual
        ), f"expected != actual,\n actual:{actual}\nexpected:{expected}"

        # clean test objects
        dir_item.store.rm("/dir-0/", recursive=True)
        expected = []
        actual = dir_item.listdir()
        assert set(expected) == set(
            actual
        ), f"expected != actual,\n actual:{actual}\nexpected:{expected}"
        dir_item.store.rm("/not-exist-dir/", recursive=True)

    @pytest.mark.parametrize("redis_endpoint", redis_endpoint)
    def test_wrong_credential_rm(self, use_datastore_profile, redis_endpoint):
        if use_datastore_profile:
            profile = DatastoreProfileRedis(
                name=self.profile_name,
                endpoint_url=redis_endpoint,
                username="not_exist",
                password="not_exist",
            )
            register_temporary_client_datastore_profile(profile)
        secrets = (
            {"REDIS_USER": "not_exist", "REDIS_PASSWORD": "not_exist"}
            if not use_datastore_profile
            else {}
        )
        data_item = mlrun.run.get_dataitem(
            redis_endpoint + "/fake_file.txt", secrets=secrets
        )
        with pytest.raises(AuthenticationError):
            data_item.delete()

    def test_empty_credential_rm(self, use_datastore_profile):
        if not use_datastore_profile:
            pytest.skip("relevant for datastore profile only")
        with pytest.raises(ValidationError):
            DatastoreProfileRedis(
                name=self.profile_name,
            )
