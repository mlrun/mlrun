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
import uuid
from urllib.parse import urlparse
from warnings import warn

import dask.dataframe as dd
import pandas as pd
import pytest
from v3io.dataplane.response import HttpResponseError

import mlrun.datastore
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestV3ioDataStore(TestMLRunSystem):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.test_file_path = str(cls.get_assets_path() / "test.txt")
        test_parquet_path = str(cls.get_assets_path() / "testdata_short.parquet")
        test_csv_path = str(cls.get_assets_path() / "testdata_short.csv")
        test_json_path = str(cls.get_assets_path() / "testdata_short.json")
        cls.df_paths = {
            "parquet": test_parquet_path,
            "csv": test_csv_path,
            "json": test_json_path,
        }
        test_additional_parquet_path = str(
            cls.get_assets_path() / "additional_data.parquet"
        )
        test_additional_csv_path = str(cls.get_assets_path() / "additional_data.csv")
        cls.additional_df_paths = {
            "parquet": test_additional_parquet_path,
            "csv": test_additional_csv_path,
        }
        with open(cls.test_file_path, "r") as f:
            cls.test_string = f.read()
        cls.test_dir_path = "/bigdata/v3io_tests"
        cls.v3io_test_dir_url = "v3io://" + cls.test_dir_path

    @classmethod
    def teardown_class(cls):
        dir_data_item = mlrun.get_dataitem(cls.v3io_test_dir_url)
        try:
            dir_data_item.delete(recursive=True)
        except Exception:
            pass
        super().teardown_class()

    def setup_method(self, method):
        self.object_dir_url = f"{self.v3io_test_dir_url}/directory-{uuid.uuid4()}"
        super().setup_method(method)

    @staticmethod
    def _skip_set_environment():
        return True

    def _get_data_item(self, secrets={}, file_extension="txt"):
        object_url = f"{self.object_dir_url}/file_{uuid.uuid4()}.{file_extension}"
        return mlrun.run.get_dataitem(object_url, secrets=secrets), object_url

    def _setup_df_dir(self, first_file_path, second_file_path, file_extension):
        dataitem_url = f"{self.object_dir_url}/df_{uuid.uuid4()}.{file_extension}"

        uploaded_data_item = mlrun.run.get_dataitem(dataitem_url)
        uploaded_data_item.upload(first_file_path)

        dataitem_url = f"{self.object_dir_url}/df_{uuid.uuid4()}.{file_extension}"

        uploaded_data_item = mlrun.run.get_dataitem(dataitem_url)
        uploaded_data_item.upload(second_file_path)

    def test_v3io_large_object_upload(self, tmp_path):
        tempfile_1_path = os.path.join(tmp_path, "tempfile_1")
        tempfile_2_path = os.path.join(tmp_path, "tempfile_2")
        cmp_command = ["cmp", tempfile_1_path, tempfile_2_path]

        with open(tempfile_1_path, "wb") as f:
            file_size = 20 * 1024 * 1024  # 20MB
            f.truncate(file_size)
            r = random.Random(123)
            for i in range(min(100, file_size)):
                offset = r.randint(0, file_size - 1)
                f.seek(offset)
                f.write(bytearray([i]))
        data_item, object_url = self._get_data_item()

        try:
            self._logger.debug(
                "Exercising the DataItem upload flow",
                tempfile_1_path=tempfile_1_path,
                tempfile_2_path=tempfile_2_path,
            )
            data_item.upload(tempfile_1_path)
            data_item.download(tempfile_2_path)

            cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
            stdout, stderr = cmp_process.communicate()
            assert (
                cmp_process.returncode == 0
            ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"

            # Do the test again, this time exercising the v3io datastore _upload() loop
            self._logger.debug("Exercising the v3io _upload() loop")
            os.remove(tempfile_2_path)
            object_path = urlparse(object_url).path
            data_item.store._upload(
                object_path, tempfile_1_path, max_chunk_size=100 * 1024
            )

            self._logger.debug("Downloading the object")
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
        data_item, object_url = self._get_data_item()
        object_path = urlparse(object_url).path
        # Exercise the DataItem put flow

        data_item.put(generated_buffer)
        returned_buffer = data_item.get()
        assert returned_buffer == generated_buffer

        data_item.store._put(object_path, generated_buffer, max_chunk_size=100 * 1024)
        returned_buffer = data_item.get()
        assert returned_buffer == generated_buffer

    def test_put_get_and_download(self):
        data_item, _ = self._get_data_item()
        data_item.put(self.test_string)
        response = data_item.get()
        assert response.decode() == self.test_string
        response = data_item.get(offset=20)
        assert response.decode() == self.test_string[20:]
        response = data_item.get(size=20)
        assert response.decode() == self.test_string[:20]
        response = data_item.get(offset=20, size=0)
        assert response.decode() == self.test_string[20:]
        response = data_item.get(offset=20, size=10)
        assert response.decode() == self.test_string[20:30]

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
            data_item.download(temp_file.name)
            content = temp_file.read()
            assert content == self.test_string

    def test_stat(self):
        data_item, _ = self._get_data_item()
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    def test_list_dir(self):
        dir_base_item = mlrun.datastore.store_manager.object(self.object_dir_url)
        file_item = mlrun.datastore.store_manager.object(
            self.object_dir_url + "/test_file.txt"
        )
        file_item_deep = mlrun.datastore.store_manager.object(
            self.object_dir_url + "/test_dir/test_file.txt"
        )
        try:
            file_item.put("test")
            file_item_deep.put("test")
            actual_dir_content = dir_base_item.listdir()
            assert actual_dir_content == ["test_dir/", "test_file.txt"]
        finally:
            dir_base_item.delete()

    def test_upload(self):
        data_item, _ = self._get_data_item()
        data_item.upload(self.test_file_path)
        response = data_item.get()
        assert response.decode() == self.test_string

    def test_rm(self):
        data_item, _ = self._get_data_item()
        data_item.upload(self.test_file_path)
        data_item.stat()
        data_item.delete()
        with pytest.raises(HttpResponseError) as http_error:
            data_item.stat()
        assert "Request failed with status 404" in str(http_error.value)
        # folder deletion:
        url = f"{self.object_dir_url}/test_directory/file.txt"
        data_item = mlrun.get_dataitem(url=url)
        data_item.upload(self.test_file_path)
        dir_data_item = mlrun.get_dataitem(url=os.path.dirname(url))
        dir_data_item.delete(recursive=True)
        with pytest.raises(HttpResponseError) as http_error_dir:
            dir_data_item.stat()
        assert "Request failed with status 404" in str(http_error_dir.value)

    @pytest.mark.parametrize(
        "file_extension,args, reader",
        [
            (
                "parquet",
                {},
                pd.read_parquet,
            ),
            ("csv", {}, pd.read_csv),
            ("json", {"orient": "records", "lines": True}, pd.read_json),
        ],
    )
    def test_as_df(
        self,
        file_extension: str,
        args: dict,
        reader: callable,
    ):
        local_file_path = self.df_paths[file_extension]
        source = reader(local_file_path, **args)
        source["date_of_birth"] = pd.to_datetime(source["date_of_birth"])
        dataitem, _ = self._get_data_item(file_extension=file_extension)
        dataitem.upload(local_file_path)
        response = dataitem.as_df(time_column="date_of_birth", **args)
        assert source.equals(response)

    @pytest.mark.parametrize(
        "file_extension,args, reader",
        [
            (
                "parquet",
                {},
                dd.read_parquet,
            ),
            ("csv", {}, dd.read_csv),
            ("json", {"orient": "records", "lines": True}, dd.read_json),
        ],
    )
    def test_as_df_dd(
        self,
        file_extension: str,
        args: dict,
        reader: callable,
    ):
        local_file_path = self.df_paths[file_extension]
        source = reader(local_file_path, **args)
        source["date_of_birth"] = dd.to_datetime(source["date_of_birth"])
        dataitem, _ = self._get_data_item(file_extension=file_extension)
        dataitem.upload(local_file_path)
        response = dataitem.as_df(time_column="date_of_birth", df_module=dd, **args)
        #  failed in json and csv becuase as_df return generator...
        # assert dd.assert_eq(source, response) # failed in parquet because:
        if source["date_of_birth"].dtype != response["date_of_birth"].dtype:
            warn("datetime types are different between source and respond", UserWarning)
        assert dd.assert_eq(
            source.drop("date_of_birth", axis=1), response.drop("date_of_birth", axis=1)
        )

    @pytest.mark.parametrize(
        "file_extension, reader",
        [
            (
                "parquet",
                pd.read_parquet,
            ),
            ("csv", pd.read_csv),
        ],
    )
    def test_check_read_df_dir(
        self,
        file_extension: str,
        reader: callable,
    ):
        first_file_path = self.df_paths[file_extension]
        second_file_path = self.additional_df_paths[file_extension]
        self._setup_df_dir(
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
        )

        dir_data_item = mlrun.run.get_dataitem(self.object_dir_url)
        response_df = (
            dir_data_item.as_df(format=file_extension, time_column="date_of_birth")
            .sort_values("id")
            .reset_index(drop=True)
        )
        df = reader(first_file_path)
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
        additional_df = reader(second_file_path)
        additional_df["date_of_birth"] = pd.to_datetime(additional_df["date_of_birth"])
        appended_df = (
            pd.concat([df, additional_df], axis=0)
            .sort_values("id")
            .reset_index(drop=True)
        )
        assert response_df.equals(appended_df)

    @pytest.mark.parametrize(
        "file_extension, reader",
        [
            (
                "parquet",
                pd.read_parquet,
            ),
            ("csv", pd.read_csv),
        ],
    )
    def test_check_read_df_dir_dd(
        self,
        file_extension: str,
        reader: callable,
    ):
        first_file_path = self.df_paths[file_extension]
        second_file_path = self.additional_df_paths[file_extension]
        self._setup_df_dir(
            first_file_path=first_file_path,
            second_file_path=second_file_path,
            file_extension=file_extension,
        )

        dir_data_item = mlrun.run.get_dataitem(self.object_dir_url)
        #  csv failed because it returns a generator ...
        response_df = (
            dir_data_item.as_df(
                format=file_extension, df_module=dd, time_column="date_of_birth"
            )
            .sort_values("id")
            .reset_index(drop=True)
        )
        df = reader(first_file_path)
        df["date_of_birth"] = dd.to_datetime(df["date_of_birth"])
        additional_df = reader(second_file_path)
        additional_df["date_of_birth"] = dd.to_datetime(additional_df["date_of_birth"])
        appended_df = (
            dd.concat([df, additional_df], axis=0)
            .sort_values("id")
            .reset_index(drop=True)
        )
        assert dd.assert_eq(appended_df, response_df)
