# Copyright 2024 Iguazio
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

import uuid

import pytest

import mlrun


class TestFileStore:
    def test_put_stat_delete(self):
        key = f"/path/file_{uuid.uuid4()}.txt"
        object_url = f"memory://{key}"
        data_item = mlrun.run.get_dataitem(object_url)
        test_text = "test string"
        data_item.put(test_text)
        assert data_item.stat().size == len(test_text)
        data_item.delete()
        with pytest.raises(ValueError, match=f"item {key} not found in memory store"):
            data_item.stat()

    def test_rm_file_not_found(self):
        not_exist_url = "memory:///path/to/file/not_exist_file.txt"
        data_item = mlrun.run.get_dataitem(not_exist_url)
        data_item.delete()
