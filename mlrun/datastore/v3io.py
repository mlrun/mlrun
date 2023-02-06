# Copyright 2018 Iguazio
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

import mmap
import os
import time
from copy import deepcopy
from datetime import datetime

import fsspec
import v3io.dataplane

import mlrun

from ..platforms.iguazio import parse_path, split_path
from .base import (
    DataStore,
    FileStats,
    basic_auth_header,
    get_range,
    http_get,
    http_head,
    http_put,
    http_upload,
)

V3IO_LOCAL_ROOT = "v3io"
ONE_GB = 1024 * 1024 * 1024
ONE_MB = 1024 * 1024


class V3ioStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self.endpoint = self.endpoint or mlrun.mlconf.v3io_api

        self.headers = None
        self.secure = self.kind == "v3ios"
        if self.endpoint.startswith("https://"):
            self.endpoint = self.endpoint[len("https://") :]
            self.secure = True
        elif self.endpoint.startswith("http://"):
            self.endpoint = self.endpoint[len("http://") :]
            self.secure = False

        token = self._get_secret_or_env("V3IO_ACCESS_KEY")
        username = self._get_secret_or_env("V3IO_USERNAME")
        password = self._get_secret_or_env("V3IO_PASSWORD")

        self.auth = None
        self.token = token
        if token:
            self.headers = {"X-v3io-session-key": token}
        elif username and password:
            self.headers = basic_auth_header(username, password)

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return V3IO_LOCAL_ROOT + subpath

    @property
    def url(self):
        schema = "https" if self.secure else "http"
        return f"{schema}://{self.endpoint}"

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            import v3iofs  # noqa
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    "v3iofs or storey not installed, run pip install storey"
                ) from exc
            return None
        self._filesystem = fsspec.filesystem("v3io", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        return dict(
            v3io_access_key=self._get_secret_or_env("V3IO_ACCESS_KEY"),
            v3io_api=mlrun.mlconf.v3io_api,
        )

    def _upload(self, key: str, src_path: str, max_chunk_size: int = ONE_GB):
        """helper function for upload method, allows for controlling max_chunk_size in testing"""
        file_size = os.path.getsize(src_path)  # in bytes
        if file_size <= ONE_MB:
            http_upload(self.url + self._join(key), src_path, self.headers, None)
            return
        append_header = deepcopy(self.headers)
        append_header["Range"] = "-1"
        with open(src_path, "rb") as file_obj:
            file_offset = 0
            while file_offset < file_size:
                chunk_size = min(file_size - file_offset, max_chunk_size)
                with mmap.mmap(
                    file_obj.fileno(),
                    length=chunk_size,
                    access=mmap.ACCESS_READ,
                    offset=file_offset,
                ) as mmap_obj:
                    http_put(
                        self.url + self._join(key),
                        mmap_obj,
                        append_header if file_offset else self.headers,
                        None,
                    )
                    file_offset += chunk_size

    def upload(self, key, src_path):
        return self._upload(key, src_path)

    def get(self, key, size=None, offset=0):
        headers = self.headers
        if size or offset:
            headers = deepcopy(headers)
            headers["Range"] = get_range(size, offset)
        return http_get(self.url + self._join(key), headers)

    def _put(self, key, data, max_chunk_size: int = ONE_GB):
        """helper function for put method, allows for controlling max_chunk_size in testing"""
        buffer_size = len(data)  # in bytes
        if buffer_size <= ONE_MB:
            http_put(self.url + self._join(key), data, self.headers, None)
            return
        append_header = deepcopy(self.headers)
        append_header["Range"] = "-1"
        buffer_offset = 0
        try:
            data = memoryview(data)
        except TypeError:
            pass
        while buffer_offset < buffer_size:
            chunk_size = min(buffer_size - buffer_offset, max_chunk_size)
            http_put(
                self.url + self._join(key),
                data[buffer_offset : buffer_offset + chunk_size],
                append_header if buffer_offset else self.headers,
                None,
            )
            buffer_offset += chunk_size

    def put(self, key, data, append=False):
        return self._put(key, data)

    def stat(self, key):
        head = http_head(self.url + self._join(key), self.headers)
        size = int(head.get("Content-Length", "0"))
        datestr = head.get("Last-Modified", "0")
        modified = time.mktime(
            datetime.strptime(datestr, "%a, %d %b %Y %H:%M:%S %Z").timetuple()
        )
        return FileStats(size, modified)

    def listdir(self, key):
        v3io_client = v3io.dataplane.Client(endpoint=self.url, access_key=self.token)
        container, subpath = split_path(self._join(key))
        if not subpath.endswith("/"):
            subpath += "/"

        # without the trailing slash
        subpath_length = len(subpath) - 1

        try:
            response = v3io_client.get_container_contents(
                container=container,
                path=subpath,
                get_all_attributes=False,
                directories_only=False,
            )
        except RuntimeError as exc:
            if "Permission denied" in str(exc):
                raise mlrun.errors.MLRunAccessDeniedError(
                    f"Access denied to path: {key}"
                ) from exc
            raise

        # todo: full = key, size, last_modified
        dir_content = [
            dir.prefix[subpath_length:] for dir in response.output.common_prefixes
        ]
        obj_content = [obj.key[subpath_length:] for obj in response.output.contents]
        return dir_content + obj_content

    def rm(self, path, recursive=False, maxdepth=None):
        """Recursive rm file/folder
        Workaround for v3io-fs not supporting recursive directory removal"""

        file_system = self.get_filesystem()
        if isinstance(path, str):
            path = [path]
        maxdepth = maxdepth if not maxdepth else maxdepth - 1
        to_rm = set()
        for p in path:
            _, p = parse_path(p, suffix="")
            p = "/" + p
            if recursive:
                find_out = file_system.find(
                    p, maxdepth=maxdepth, withdirs=True, detail=True
                )
                rec = set(
                    sorted(
                        [
                            f["name"] + ("/" if f["type"] == "directory" else "")
                            for f in find_out.values()
                        ]
                    )
                )
                to_rm |= rec
            if p not in to_rm and (recursive is False or file_system.exists(p)):
                p = p + ("/" if file_system.isdir(p) else "")
                to_rm.add(p)
        for p in reversed(list(sorted(to_rm))):
            file_system.rm_file(p)
