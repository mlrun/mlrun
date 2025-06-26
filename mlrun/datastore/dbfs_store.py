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

import pathlib
from typing import Optional

from fsspec.implementations.dbfs import DatabricksFile, DatabricksFileSystem
from fsspec.registry import get_filesystem_class

import mlrun.errors

from .base import DataStore, FileStats, make_datastore_schema_sanitizer


class DatabricksFileBugFixed(DatabricksFile):
    """Overrides DatabricksFile to add the following fix: https://github.com/fsspec/filesystem_spec/pull/1278"""

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


class DatabricksFileSystemDisableCache(DatabricksFileSystem):
    root_marker = "/"
    protocol = "dbfs"

    def _open(self, path, mode="rb", block_size="default", **kwargs):
        """
        Overwrite the base class method to make sure to create a DBFile.
        All arguments are copied from the base method.

        Only the default blocksize is allowed.
        """
        return DatabricksFileBugFixed(
            self, path, mode=mode, block_size=block_size, **kwargs
        )

    #  _ls_from_cache is not working properly, so we disable it.
    def _ls_from_cache(self, path):
        pass


# dbfs objects will be represented with the following URL: dbfs://<path>
class DBFSStore(DataStore):
    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        filesystem_class = get_filesystem_class(protocol=self.kind)
        if not self._filesystem:
            self._filesystem = make_datastore_schema_sanitizer(
                cls=filesystem_class,
                using_bucket=False,
                **self.get_storage_options(),
            )
        return self._filesystem

    def get_storage_options(self):
        res = dict(
            token=self._get_secret_or_env("DATABRICKS_TOKEN"),
            instance=self._get_secret_or_env("DATABRICKS_HOST"),
        )
        return self._sanitize_storage_options(res)

    def _verify_filesystem_and_key(self, key: str):
        if not self.filesystem:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Performing actions on data-item without a valid filesystem"
            )
        if not key.startswith("/"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid key parameter - key must start with '/'"
            )

    def get(self, key: str, size=None, offset=0) -> bytes:
        self._verify_filesystem_and_key(key)
        if size is not None and size < 0:
            raise mlrun.errors.MLRunInvalidArgumentError("size cannot be negative")
        if offset is None:
            raise mlrun.errors.MLRunInvalidArgumentError("offset cannot be None")
        start = offset or None
        end = offset + size if size else None
        return self.filesystem.cat_file(key, start=start, end=end)

    def put(self, key, data, append=False):
        self._verify_filesystem_and_key(key)
        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Databricks file system"
            )
        #  can not use append mode because it overrides data.
        data, mode = self._prepare_put_data(data, append)
        with self.filesystem.open(key, mode) as f:
            f.write(data)

    def upload(self, key: str, src_path: str):
        self._verify_filesystem_and_key(key)
        self.filesystem.put_file(src_path, key, overwrite=True)

    def stat(self, key: str):
        self._verify_filesystem_and_key(key)
        file = self.filesystem.stat(key)
        if file["type"] == "file":
            size = file["size"]
        elif file["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        return FileStats(size, None)

    def listdir(self, key: str):
        """
        Basic ls of file/dir - without recursion.
        """
        self._verify_filesystem_and_key(key)
        if self.filesystem.isfile(key):
            return key
        remote_path = f"{key}/*"
        files = self.filesystem.glob(remote_path)
        # Get only the files and directories under key path, without the key path itself.
        # for example in a filesystem that has this path: /test_mlrun_dbfs_objects/test.txt
        # listdir with the input /test_mlrun_dbfs_objects as a key will return ['test.txt'].
        files = [pathlib.Path(file).name for file in files if "/" in file]
        return files

    def rm(self, path, recursive=False, maxdepth=None):
        if maxdepth:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "dbfs file system does not support maxdepth option in rm function"
            )
        self.filesystem.rm(path=path, recursive=recursive)
