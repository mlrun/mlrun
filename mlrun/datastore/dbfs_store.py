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

from fsspec.implementations.dbfs import DatabricksFile, DatabricksFileSystem

import mlrun.errors

from .base import DataStore, FileStats

# dbfs objects will be represented with the following URL: dbfs://<path>


class DatabricksFileRangeFix(DatabricksFile):
    def _upload_chunk(self, final=False):
        """Internal function to add a chunk of data to a started upload"""
        self.buffer.seek(0)
        data = self.buffer.getvalue()

        data_chunks = [
            data[start:end] for start, end in self._to_sized_blocks(end=len(data))
        ]

        for data_chunk in data_chunks:
            self.fs._add_data(handle=self.handle, data=data_chunk)

        if final:
            self.fs._close_handle(handle=self.handle)
            return True

    def _fetch_range(self, start, end):
        """Internal function to download a block of data"""
        return_buffer = b""
        for chunk_start, chunk_end in self._to_sized_blocks(start, end):
            return_buffer += self.fs._get_data(
                path=self.path, start=chunk_start, end=chunk_end
            )

        return return_buffer

    def _to_sized_blocks(self, start=0, end=100):
        """Helper function to split a range from 0 to total_length into blocksizes"""
        for data_chunk in range(start, end, self.blocksize):
            data_start = data_chunk
            data_end = min(end, data_chunk + self.blocksize)
            yield data_start, data_end


class DatabricksFileSystemRangeFix(DatabricksFileSystem):
    def _open(self, path, mode="rb", block_size="default", **kwargs):
        """
        Overwrite the base class method to make sure to create a DBFile.
        All arguments are copied from the base method.

        Only the default blocksize is allowed.
        """
        return DatabricksFileRangeFix(
            self, path, mode=mode, block_size=block_size, **kwargs
        )


class DBFSStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self.get_filesystem(silent=False)

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if not self._filesystem:
            #  self._filesystem = fsspec.filesystem("dbfs", **self.get_storage_options())
            self._filesystem = DatabricksFileSystemRangeFix(
                **self.get_storage_options()
            )
        return self._filesystem

    def get_storage_options(self):
        return dict(
            token=self._get_secret_or_env("DATABRICKS_TOKEN"), instance=self.endpoint
        )

    def _prepare_path_and_verify_filesystem(self, key: str):
        if not self._filesystem:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Performing actions on data-item without a valid filesystem"
            )
        if not key.startswith("/"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid key parameter - key must start with '/'"
            )
        return key

    def get(self, key: str, size=None, offset=0) -> bytes:
        key = self._prepare_path_and_verify_filesystem(key)
        if size is not None and size <= 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "size cannot be negative or zero"
            )
        if not size and not offset:
            return self._filesystem.cat_file(key)
        end = offset + size if size is not None else None
        return self._filesystem.cat_file(key, start=offset, end=end)

    def put(self, key, data, append=False):

        key = self._prepare_path_and_verify_filesystem(key)
        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Databricks file system"
            )
        #  can not use append mode because it overrides data.
        mode = "w"
        if isinstance(data, bytes):
            mode += "b"
        elif not isinstance(data, str):
            raise TypeError(f"Unknown data type {type(data)}")
        with self._filesystem.open(key, mode) as f:
            f.write(data)

    def upload(self, key: str, src_path: str):
        key = self._prepare_path_and_verify_filesystem(key)
        self._filesystem.put_file(src_path, key, overwrite=True)

    def stat(self, key: str):
        key = self._prepare_path_and_verify_filesystem(key)
        files = self._filesystem.ls(key, detail=True)
        if len(files) == 1 and files[0]["type"] == "file":
            size = files[0]["size"]
        elif len(files) == 1 and files[0]["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        else:
            raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, None)

    def listdir(self, key: str):
        """
        Basic ls of file/dir - without recursion.
        """
        key = self._prepare_path_and_verify_filesystem(key)
        if self._filesystem.isfile(key):
            return key
        remote_path = f"{key}/*"
        files = self._filesystem.glob(remote_path)
        key_length = len(key)
        #  Get only the files and directories under key path, without the key path itself.
        # for example for /test_mlrun_dbfs_objects/test.txt the function will return ['test.txt'].
        files = [
            file.split("/", 1)[1][key_length:]
            for file in files
            if len(file.split("/")) > 1
        ]
        return files

    def supports_isdir(self):
        return False

    def rm(self, path, recursive=False, maxdepth=None):
        if maxdepth:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "dbfs file system does not support maxdepth option in rm function."
            )
        self.get_filesystem().rm(path=path, recursive=recursive)

    def as_df(
        self,
        url,
        subpath,
        columns=None,
        df_module=None,
        format="",
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        return super().as_df(
            url=subpath,
            subpath="",
            columns=columns,
            df_module=df_module,
            format=format,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            **kwargs,
        )
