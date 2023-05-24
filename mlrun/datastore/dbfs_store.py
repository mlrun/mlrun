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

import fsspec

import mlrun.errors
from mlrun.datastore.helpers import ONE_GB, ONE_MB
from .base import DataStore, FileStats

# dbfs objects will be represented with the following URL: dbfs://<path>


class DBFSStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        if self.endpoint.startswith("https://"):
            self.endpoint.replace("https://", "")
        if self.endpoint.startswith("http://"):
            self.endpoint.replace("http://", "")
        self.get_filesystem(silent=False)

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if not self._filesystem:
            self._filesystem = fsspec.filesystem("dbfs", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        return dict(
            token=self._get_secret_or_env("DATABRICKS_TOKEN"), instance=self.endpoint
        )

    def path_and_system_validator(self, key: str):
        if not self._filesystem:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Performing actions on data-item without a valid filesystem"
            )
        if not key.startswith("/"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid path(key attribute) - path must start with '/'"
            )

    def get(self, key: str, size=None, offset=0) -> bytes:
        self.path_and_system_validator(key)
        size_less_than_one_exception = "negative or zero size argument 1 is invalid."
        if offset:
            if size is None:
                size = ONE_MB  # The maximum number of allowed bytes to read is 1MB.
            elif size <= 0:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    size_less_than_one_exception
                )
            end = offset + size
            return self._filesystem._get_data(key, start=offset, end=end)
        elif size is not None:
            if size <= 0:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    size_less_than_one_exception
                )
            else:
                end = offset + size
                return self._filesystem._get_data(key, start=offset, end=end)
        else:
            return self._filesystem.cat_file(key)

    def put(self, key, data, append=False):

        self.path_and_system_validator(key)
        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Databricks file system!"
            )
        #  can not use append mode because it overrides data.
        mode = "w"
        if isinstance(data, bytes):
            mode += "b"
        elif not isinstance(data, str):
            raise TypeError(
                "Data type unknown.  Unable to put in Databricks file system!"
            )
        with self._filesystem.open(key, mode) as f:
            f.write(data)

    def upload(self, key: str, src_path: str):
        self.path_and_system_validator(key)
        self._filesystem.put_file(src_path, key, overwrite=True)

    def stat(self, key: str):
        self.path_and_system_validator(key)
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
        self.path_and_system_validator(key)
        if self._filesystem.isfile(key):
            return key
        remote_path = f"{key}/*"
        files = self._filesystem.glob(remote_path)
        key_length = len(key)
        files = [
            f.split("/", 1)[1][key_length:] for f in files if len(f.split("/")) > 1
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
