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
import random
import subprocess
import tempfile

import pytest

import mlrun.datastore
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestV3ioDataStore(TestMLRunSystem):
    @staticmethod
    def _skip_set_environment():
        return True

    def test_v3io_large_object_upload(self):

        dir = tempfile.TemporaryDirectory()
        tempfile_1_path = os.path.join(dir.name, "tempfile_1")
        tempfile_2_path = os.path.join(dir.name, "tempfile_2")
        cmp_command = ["cmp", tempfile_1_path, tempfile_2_path]

        with open(tempfile_1_path, "wb") as f:
            file_size = 20 * 1024 * 1024  # 20MB
            f.truncate(file_size)
            r = random.Random(123)
            for i in range(min(100, file_size)):
                offset = r.randint(0, file_size - 1)
                f.seek(offset)
                f.write(bytearray([i]))
        object_path = "/bigdata/test_v3io_large_object_upload"
        v3io_object_url = "v3io://" + object_path

        data_item = mlrun.datastore.store_manager.object(v3io_object_url)

        try:
            # Exercise the DataItem upload flow
            data_item.upload(tempfile_1_path)
            data_item.download(tempfile_2_path)

            cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
            stdout, stderr = cmp_process.communicate()
            assert (
                cmp_process.returncode == 0
            ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"

            # Do the test again, this time exercising the v3io datastore _upload() loop
            os.remove(tempfile_2_path)

            data_item.store._upload(
                object_path, tempfile_1_path, max_chunk_size=100 * 1024
            )
            data_item.download(tempfile_2_path)

            cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
            stdout, stderr = cmp_process.communicate()
            assert (
                cmp_process.returncode == 0
            ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"

        finally:
            # cleanup (local files are cleaned by the TempDir destructor)
            data_item.delete()

    def test_v3io_large_object_put(self):
        file_size = 20 * 1024 * 1024  # 20MB
        generated_buffer = bytearray(os.urandom(file_size))
        object_path = "/bigdata/test_v3io_large_object_put"
        v3io_object_url = "v3io://" + object_path
        data_item = mlrun.datastore.store_manager.object(v3io_object_url)
        try:
            # Exercise the DataItem put flow
            data_item.put(generated_buffer)
            returned_buffer = data_item.get()
            assert returned_buffer == generated_buffer

            data_item.store._put(
                object_path, generated_buffer, max_chunk_size=100 * 1024
            )
            returned_buffer = data_item.get()
            assert returned_buffer == generated_buffer

        finally:
            data_item.delete()

    def test_list_dir(self):
        dir_base_path = "/bigdata/test_base_dir/"
        v3io_dir_url = "v3io://" + dir_base_path
        dir_base_item = mlrun.datastore.store_manager.object(v3io_dir_url)
        file_item = mlrun.datastore.store_manager.object(v3io_dir_url + "test_file")
        file_item_deep = mlrun.datastore.store_manager.object(
            v3io_dir_url + "test_dir/test_file"
        )
        try:
            file_item.put("test")
            file_item_deep.put("test")
            actual_dir_content = dir_base_item.listdir()
            assert actual_dir_content == ["test_dir/", "test_file"]
        finally:
            dir_base_item.delete()
