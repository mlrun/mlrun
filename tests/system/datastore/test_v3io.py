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
import os.path
import subprocess
import tempfile
import time
import uuid
from urllib.parse import urlparse

import dask.dataframe as dd
import pandas as pd
import pytest

import mlrun.datastore
from mlrun.datastore.datastore_profile import (
    DatastoreProfileV3io,
    register_temporary_client_datastore_profile,
)
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestV3ioDataStore(TestMLRunSystem):
    test_dir = "/bigdata/v3io_tests"
    test_dir_url = f"v3io://{test_dir}"
    run_dir = f"{test_dir}/run_{uuid.uuid4()}"
    profile_name = "v3io_ds_profile"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.token = os.environ.get("V3IO_ACCESS_KEY")
        assets_path = str(cls.get_assets_path())
        cls.test_file_path = os.path.join(assets_path, "test.txt")
        test_parquet_path = os.path.join(assets_path, "test_data.parquet")
        test_csv_path = os.path.join(assets_path, "test_data.csv")
        test_json_path = os.path.join(assets_path, "test_data.json")
        cls.df_paths = {
            "parquet": test_parquet_path,
            "csv": test_csv_path,
            "json": test_json_path,
        }
        test_additional_parquet_path = os.path.join(
            assets_path, "additional_data.parquet"
        )
        test_additional_csv_path = os.path.join(assets_path, "additional_data.csv")
        cls.additional_df_paths = {
            "parquet": test_additional_parquet_path,
            "csv": test_additional_csv_path,
        }
        with open(cls.test_file_path) as f:
            cls.test_string = f.read()
        cls.profile = DatastoreProfileV3io(
            name=cls.profile_name, v3io_access_key=cls.token
        )

    @classmethod
    def teardown_class(cls):
        dir_data_item = mlrun.get_dataitem(cls.test_dir_url)
        try:
            dir_data_item.delete(recursive=True)
        except Exception:
            cls._logger.warning(
                f"failed to delete test directory {cls.test_dir_url} in test_v3io.py."
            )
        super().teardown_class()

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        prefix_path = (
            f"ds://{self.profile_name}" if use_datastore_profile else "v3io://"
        )
        mlrun.datastore.store_manager.reset_secrets()
        self.run_dir_url = f"{prefix_path}{self.run_dir}"
        object_file = f"/file_{uuid.uuid4()}.txt"
        self._object_url = f"{self.run_dir_url}{object_file}"
        register_temporary_client_datastore_profile(self.profile)

        # We give priority to profiles, then to secrets, and finally to environment variables.
        # We want to ensure that we test these priorities in the correct order.
        if use_datastore_profile:
            os.environ["V3IO_ACCESS_KEY"] = "wrong_token"
        else:
            os.environ["V3IO_ACCESS_KEY"] = self.token

    def teardown_method(self, method):
        os.environ["V3IO_ACCESS_KEY"] = self.token
        super().teardown_method(method=method)

    @staticmethod
    def _skip_set_environment():
        return True

    def test_v3io_large_object_upload(self, tmp_path):
        tempfile_1_path = os.path.join(tmp_path, "tempfile_1")
        tempfile_2_path = os.path.join(tmp_path, "tempfile_2")
        cmp_command = ["cmp", tempfile_1_path, tempfile_2_path]
        first_start_time = time.time()
        with open(tempfile_1_path, "wb") as f:
            file_size = 20 * 1024 * 1024  # 20MB
            data = os.urandom(file_size)
            f.write(data)
        data_item = mlrun.run.get_dataitem(self._object_url)
        self._logger.debug(
            f"test_v3io_large_object_upload - finished to write locally in {time.time() - first_start_time} seconds"
        )
        self._logger.debug(
            "Exercising the DataItem upload flow",
            tempfile_1_path=tempfile_1_path,
            tempfile_2_path=tempfile_2_path,
        )
        start_time = time.time()
        data_item.upload(tempfile_1_path)
        self._logger.debug(
            f"test_v3io_large_object_upload - finished to upload in {time.time() - start_time} seconds"
        )
        start_time = time.time()
        data_item.download(tempfile_2_path)
        self._logger.debug(
            f"test_v3io_large_object_upload - finished to download locally in {time.time() - start_time} seconds"
        )
        start_time = time.time()
        cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
        stdout, stderr = cmp_process.communicate()
        assert (
            cmp_process.returncode == 0
        ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"
        self._logger.debug(
            f"test_v3io_large_object_upload - finished cmp 1 in {time.time() - start_time} seconds"
        )
        # Do the test again, this time exercising the v3io datastore _upload() loop
        self._logger.debug("Exercising the v3io _upload() loop")
        os.remove(tempfile_2_path)
        object_path = urlparse(self._object_url).path
        start_time = time.time()
        data_item.store._upload(object_path, tempfile_1_path, max_chunk_size=100 * 1024)
        self._logger.debug(
            f"test_v3io_large_object_upload - finished to upload with store directly in"
            f" {time.time() - start_time} seconds"
        )
        self._logger.debug("Downloading the object")
        start_time = time.time()
        data_item.download(tempfile_2_path)
        self._logger.debug(
            f"test_v3io_large_object_upload - finished to download in the second time in"
            f" {time.time() - start_time} seconds"
        )
        start_time = time.time()
        cmp_process = subprocess.Popen(cmp_command, stdout=subprocess.PIPE)
        stdout, stderr = cmp_process.communicate()
        assert (
            cmp_process.returncode == 0
        ), f"stdout = {stdout}, stderr={stderr}, returncode={cmp_process.returncode}"
        self._logger.debug(
            f"test_v3io_large_object_upload - finished cmp 2 in {time.time() - start_time} seconds"
        )
        self._logger.debug(
            f"total time of test_v3io_large_object_upload {time.time() - first_start_time}"
        )

    def test_v3io_large_object_put(self):
        file_size = 20 * 1024 * 1024  # 20MB
        generated_buffer = bytearray(os.urandom(file_size))
        data_item = mlrun.run.get_dataitem(self._object_url)
        object_path = urlparse(self._object_url).path

        first_start_time = time.time()
        data_item.put(generated_buffer)
        self._logger.debug(
            f"test_v3io_large_object_put: put finished in : {time.time() - first_start_time} seconds"
        )
        start_time = time.time()
        returned_buffer = data_item.get()
        self._logger.debug(
            f"test_v3io_large_object_put: first get finished in : {time.time() - start_time} seconds"
        )
        assert returned_buffer == generated_buffer
        start_time = time.time()
        data_item.store._put(object_path, generated_buffer, max_chunk_size=100 * 1024)
        self._logger.debug(
            f"test_v3io_large_object_put: store put finished in : {time.time() - start_time} seconds"
        )
        start_time = time.time()
        returned_buffer = data_item.get()
        self._logger.debug(
            f"test_v3io_large_object_put: second get finished in : {time.time() - start_time} seconds"
        )
        assert returned_buffer == generated_buffer
        self._logger.debug(
            f"test_v3io_large_object_put: total time: {time.time() - first_start_time} seconds"
        )

    @pytest.mark.parametrize("use_secrets_as_parameters", [True, False])
    def test_put_get_and_download(
        self, use_datastore_profile, use_secrets_as_parameters
    ):
        secrets = {}
        if use_secrets_as_parameters:
            os.environ["V3IO_ACCESS_KEY"] = "wrong_token"
            # Verify that we are using the correct profile secret by deliberately setting
            # an incorrect token as the secret or env. We expect that the correct token,
            # which is saved in the datastore profile, will be utilized.
            secrets = (
                {"V3IO_ACCESS_KEY": "wrong_token"}
                if use_datastore_profile
                else {"V3IO_ACCESS_KEY": self.token}
            )

        data_item = mlrun.run.get_dataitem(self._object_url, secrets=secrets)
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

        # append=True test:
        data_item.put(self.test_string, append=True)
        response = data_item.get()
        assert response.decode() == self.test_string + self.test_string

    def test_stat(self):
        data_item = mlrun.run.get_dataitem(self._object_url)
        data_item.put(self.test_string)
        stat = data_item.stat()
        assert stat.size == len(self.test_string)

    def test_list_dir(self):
        dir_base_item = mlrun.datastore.store_manager.object(self.run_dir_url)
        filename = f"test_file_{uuid.uuid4()}.txt"
        file_item = mlrun.datastore.store_manager.object(
            f"{self.run_dir_url}/{filename}"
        )
        file_item_deep = mlrun.datastore.store_manager.object(
            f"{self.run_dir_url}/test_dir/test_file_{uuid.uuid4()}.txt"
        )
        file_item.put("test")
        file_item_deep.put("test")
        actual_dir_content = dir_base_item.listdir()
        assert all(item in actual_dir_content for item in ["test_dir/", filename])

    def test_upload(self):
        data_item = mlrun.run.get_dataitem(self._object_url)
        data_item.upload(self.test_file_path)
        response = data_item.get()
        assert response.decode() == self.test_string

    def test_rm(self):
        data_item = mlrun.run.get_dataitem(self._object_url)
        data_item.upload(self.test_file_path)
        data_item.stat()
        data_item.delete()
        with pytest.raises(
            mlrun.errors.MLRunNotFoundError, match="Request failed with status 404"
        ):
            data_item.stat()

    @pytest.mark.parametrize(
        "file_format, pd_reader, dd_reader, reader_args",
        [
            ("parquet", pd.read_parquet, dd.read_parquet, {}),
            ("csv", pd.read_csv, dd.read_csv, {}),
            ("json", pd.read_json, dd.read_json, {"orient": "records"}),
        ],
    )
    def test_as_df(
        self,
        file_format: str,
        pd_reader: callable,
        dd_reader: callable,
        reader_args: dict,
    ):
        filename = f"df_{uuid.uuid4()}.{file_format}"
        dataframe_url = f"{self.run_dir_url}/{filename}"
        local_file_path = os.path.join(self.assets_path, f"test_data.{file_format}")

        source = pd_reader(local_file_path, **reader_args)
        upload_data_item = mlrun.run.get_dataitem(dataframe_url)
        upload_data_item.upload(local_file_path)
        response = upload_data_item.as_df(**reader_args)
        pd.testing.assert_frame_equal(source, response)

        # dask
        source = dd_reader(local_file_path, **reader_args)
        response = upload_data_item.as_df(**reader_args, df_module=dd)
        dd.assert_eq(source, response)

    @pytest.mark.parametrize(
        "file_format, pd_reader, dd_reader, reset_index",
        [
            ("parquet", pd.read_parquet, dd.read_parquet, False),
            ("csv", pd.read_csv, dd.read_csv, True),
        ],
    )
    def test_as_df_directory(self, file_format, pd_reader, dd_reader, reset_index):
        dataframes_dir = f"/{file_format}_{uuid.uuid4()}"
        dataframes_url = f"{self.run_dir_url}{dataframes_dir}"
        df1_path = os.path.join(self.assets_path, f"test_data.{file_format}")
        df2_path = os.path.join(self.assets_path, f"additional_data.{file_format}")

        # upload
        dt1 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df1.{file_format}",
        )
        dt2 = mlrun.run.get_dataitem(
            f"{dataframes_url}/df2.{file_format}",
        )
        dt1.upload(src_path=df1_path)
        dt2.upload(src_path=df2_path)
        dt_dir = mlrun.run.get_dataitem(dataframes_url)
        df1 = pd_reader(df1_path)
        df2 = pd_reader(df2_path)
        expected_df = pd.concat([df1, df2], ignore_index=True)
        tested_df = dt_dir.as_df(format=file_format)
        if reset_index:
            tested_df = tested_df.sort_values("id").reset_index(drop=True)
        pd.testing.assert_frame_equal(tested_df, expected_df)

        # dask
        dd_df1 = dd_reader(df1_path)
        dd_df2 = dd_reader(df2_path)
        expected_dd_df = dd.concat([dd_df1, dd_df2], axis=0)
        tested_dd_df = dt_dir.as_df(format=file_format, df_module=dd)
        dd.assert_eq(tested_dd_df, expected_dd_df)
