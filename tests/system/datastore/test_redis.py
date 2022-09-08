# Copyright 2022 Iguazio
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

import pytest

import mlrun.datastore
from tests.system.base import TestMLRunSystem


@pytest.mark.skipif(
    not mlrun.mlconf.redis_url, reason="mlrun.mlconf.redis_url is not set"
)
class TestRedisDataStore(TestMLRunSystem):
    @staticmethod
    def _skip_set_environment():
        return True

    def test_redis_put_get_object(self):

        redis_path = "redis:///redis_object"
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

    def test_redis_upload_download_object(self):
        # prepare file for upload
        expected = "abcde" * 100
        with open("temp_upload", "w") as f:
            f.write(expected)

        redis_path = "redis:///redis_object"
        data_item = mlrun.datastore.store_manager.object(redis_path)

        data_item.delete()

        data_item.upload("temp_upload")
        data_item.download("temp_download")

        with open("temp_download", "r") as f:
            actual = f.read()

        data_item.delete()
        os.remove("temp_upload")
        os.remove("temp_download")

        assert expected == actual

    def test_redis_listdir(self):

        redis_path = "redis://"
        dir_path = redis_path
        expected = []
        list_dir = redis_path + "/dir-0/dir-1"

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
