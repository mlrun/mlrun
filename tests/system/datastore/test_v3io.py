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
import random
import subprocess

import pytest

import mlrun.datastore
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestV3ioDataStore(TestMLRunSystem):
    @staticmethod
    def _skip_set_environment():
        return True

    def test_v3io_large_object(self):

        with open("tempfile_1", "wb") as f:
            file_size = 20 * 1024 * 1024  # 20MB
            f.truncate(file_size)
            r = random.Random(123)
            for i in range(min(100, file_size)):
                offset = r.randint(0, file_size - 1)
                f.seek(offset)
                f.write(bytearray([i]))
        v3io_path = "v3io:///bigdata/test_v3io_large_object"
        data_item = mlrun.datastore.store_manager.object(v3io_path)

        data_item.upload("tempfile_1")
        data_item.download("tempfile_2")

        cmp_command = ["cmp", "tempfile_1", "tempfile_2"]
        cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
        stdout, stderr = cmp_process.communicate()

        data_item.delete()
        os.remove("tempfile_1")
        os.remove("tempfile_2")

        assert (
            cmp_process.returncode == 0
        ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"
