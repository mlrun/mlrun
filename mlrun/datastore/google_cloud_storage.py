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
from urllib.parse import urlparse

from gcsfs.core import GCSFileSystem

import mlrun.errors
from mlrun.utils import logger

from .base import DataStoreWithBucket, FileStats
from .datastore_profile import datastore_profile_read

# Google storage objects will be represented with the following URL: gcs://<bucket name>/<path> or gs://...


class GCSFileSystemWithDS(GCSFileSystem):
    @classmethod
    def _strip_protocol(cls, url):
        if url.startswith("ds://"):
            parsed_url = urlparse(url)
            url = parsed_url.path[1:]
        return super()._strip_protocol(url)


class GoogleCloudStorageStore(DataStoreWithBucket):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        if schema == "ds":
            if secrets:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Trying to use secret and profile at dbfs datastore."
                )
            
        # Workaround to bypass the fact that fsspec works with gcs such that credentials must be placed in a JSON
        # file, and pointed at by the GOOGLE_APPLICATION_CREDENTIALS env. variable. When passing it to runtime pods,
        # eventually we will want this to happen through a secret that is mounted as a file to the pod. For now,
        # we just read a specific env. variable, write it to a temp file and point the env variable to it.
        elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            gcp_credentials = self._get_secret_or_env("GCP_CREDENTIALS")
            if gcp_credentials:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as cred_file:
                    cred_file.write(gcp_credentials)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file.name
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
        self._filesystem = GCSFileSystemWithDS(**self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        if self.kind == "ds":
            datastore_profile = datastore_profile_read(self.name)
            google_application_credentials = (
                datastore_profile.google_application_credentials
            )
            gcp_credentials = datastore_profile.gcp_credentials
            if google_application_credentials:
                credentials_file_name = google_application_credentials
            elif gcp_credentials:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as credentials_file:
                    credentials_file.write(gcp_credentials)
                    credentials_file_name = credentials_file.name
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Datastore profile must have google_application_credentials or gcp_credentials."
                )
            return dict(token=credentials_file_name)
        return dict(token=self._get_secret_or_env("GCS_TOKEN"))

    def get(self, key, size=None, offset=0):
        _, _, path = self.get_bucket_and_key(key)
        end = offset + size if size else None
        blob = self._filesystem.cat_file(path, start=offset, end=end)
        return blob

    def put(self, key, data, append=False):
        _, _, path = self.get_bucket_and_key(key)

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
        _, _, path = self.get_bucket_and_key(key)
        self._filesystem.put_file(src_path, path, overwrite=True)

    def stat(self, key):
        _, _, path = self.get_bucket_and_key(key)

        files = self._filesystem.ls(path, detail=True)
        if len(files) == 1 and files[0]["type"] == "file":
            size = files[0]["size"]
            modified = files[0]["updated"]
        elif len(files) == 1 and files[0]["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        else:
            raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, modified)

    def listdir(self, key: str):
        bucket, _, path = self.get_bucket_and_key(key)
        if self._filesystem.isfile(path):
            return key
        remote_path = f"{path}/**"
        files = self._filesystem.glob(remote_path)
        path_length = len(path) - len(bucket)
        files = [
            f.split("/", 1)[1][path_length:] for f in files if len(f.split("/")) > 1
        ]
        return files
