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
import os.path
import tempfile

import pytest

import mlrun


@pytest.mark.parametrize(
    "prefix",
    ["", "file://"],
)
class TestFileStore:
    def test_put_stat_delete(self, prefix):
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                object_url = f"{prefix}{temp_file.name}"
                data_item = mlrun.run.get_dataitem(object_url)
                test_text = "test string"
                data_item.put(test_text)
                assert data_item.stat().size == len(test_text)
                data_item.delete()
                with pytest.raises(FileNotFoundError):
                    data_item.stat()
                assert not os.path.exists(temp_file.name)
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def test_rm_file_not_found(self, prefix):
        not_exist_url = f"{prefix}/path/to/file/not_exist_file.txt"
        data_item = mlrun.run.get_dataitem(not_exist_url)
        data_item.delete()
