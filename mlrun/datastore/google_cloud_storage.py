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
import tempfile
from pathlib import Path

import mlrun.errors
from mlrun.utils import logger

from .base import DataStore, FileStats, makeDatastoreSchemaSanitizer

# Google storage objects will be represented with the following URL: gcs://<bucket name>/<path> or gs://...


class GoogleCloudStorageStore(DataStore):
    using_bucket = True

    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)

        # Gives priority to secrets GOOGLE_APPLICATION_CREDENTIALS,
        # then secrets GCP_CREDENTIALS,
        # then environment GOOGLE_APPLICATION_CREDENTIALS,
        # and finally, environment GCP_CREDENTIALS.
        # Secrets have first priority, especially useful for profile cases.

        choose_gcp_credentials = (
            self._secrets
            and "GCP_CREDENTIALS" in self._secrets
            and "GOOGLE_APPLICATION_CREDENTIALS" not in self._secrets
        ) or ("GOOGLE_APPLICATION_CREDENTIALS" not in os.environ)

        if choose_gcp_credentials:
            # Workaround to bypass the fact that fsspec works with gcs such that credentials must be placed in a JSON
            # file, and pointed at by the GOOGLE_APPLICATION_CREDENTIALS env. variable. When passing it to runtime pods,
            # eventually we will want this to happen through a secret that is mounted as a file to the pod. For now,
            # we just read a specific env. variable, write it to a temp file and point the env variable to it.
            gcp_credentials = self._get_secret_or_env("GCP_CREDENTIALS")
            if gcp_credentials:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as cred_file:
                    cred_file.write(gcp_credentials)
                    self._secrets["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file.name
            else:
                logger.info(
                    "No GCS credentials available - auth will rely on auto-discovery of credentials"
                )

        self.get_filesystem(silent=False)

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem

        try:
            import gcsfs  # noqa
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    "Google gcsfs not installed, run pip install gcsfs"
                ) from exc
            return None
        self._filesystem = makeDatastoreSchemaSanitizer(
            gcsfs.core.GCSFileSystem,
            using_bucket=self.using_bucket,
            **self.get_storage_options(),
        )
        return self._filesystem

    def get_storage_options(self):
        return dict(token=self._get_secret_or_env("GOOGLE_APPLICATION_CREDENTIALS"))

    def _prepare_path_and_verify_filesystem(self, key):
        if not self._filesystem:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Performing actions on data-item without a valid filesystem"
            )

        key = key.strip("/")
        path = Path(self.endpoint, key).as_posix()
        return path

    def get(self, key, size=None, offset=0):
        path = self._prepare_path_and_verify_filesystem(key)

        end = offset + size if size else None
        blob = self._filesystem.cat_file(path, start=offset, end=end)
        return blob

    def put(self, key, data, append=False):
        path = self._prepare_path_and_verify_filesystem(key)

        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Google cloud storage datastore"
            )

        if isinstance(data, bytes):
            mode = "wb"
        elif isinstance(data, str):
            mode = "w"
        else:
            raise TypeError(
                "Data type unknown.  Unable to put in Google cloud storage!"
            )
        with self._filesystem.open(path, mode) as f:
            f.write(data)

    def upload(self, key, src_path):
        path = self._prepare_path_and_verify_filesystem(key)
        self._filesystem.put_file(src_path, path, overwrite=True)

    def stat(self, key):
        path = self._prepare_path_and_verify_filesystem(key)

        files = self._filesystem.ls(path, detail=True)
        if len(files) == 1 and files[0]["type"] == "file":
            size = files[0]["size"]
            modified = files[0]["updated"]
        elif len(files) == 1 and files[0]["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        else:
            raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, modified)

    def listdir(self, key):
        path = self._prepare_path_and_verify_filesystem(key)
        if self._filesystem.isfile(path):
            return key
        remote_path = f"{path}/**"
        files = self._filesystem.glob(remote_path)
        key_length = len(key)
        files = [
            f.split("/", 1)[1][key_length:] for f in files if len(f.split("/")) > 1
        ]
        return files

    def rm(self, path, recursive=False, maxdepth=None):
        path = self._prepare_path_and_verify_filesystem(path)
        self.get_filesystem().rm(path=path, recursive=recursive, maxdepth=maxdepth)
