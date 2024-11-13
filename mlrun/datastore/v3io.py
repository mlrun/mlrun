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

import time
from datetime import datetime
from typing import Optional

import fsspec
import v3io
from v3io.dataplane.response import HttpResponseError

import mlrun

from ..platforms.iguazio import parse_path, split_path
from .base import (
    DataStore,
    FileStats,
    basic_auth_header,
)

V3IO_LOCAL_ROOT = "v3io"
V3IO_DEFAULT_UPLOAD_CHUNK_SIZE = 1024 * 1024 * 10


class V3ioStore(DataStore):
    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self.endpoint = self.endpoint or mlrun.mlconf.v3io_api

        self.headers = None
        self.secure = self.kind == "v3ios"

        token = self._get_secret_or_env("V3IO_ACCESS_KEY")
        username = self._get_secret_or_env("V3IO_USERNAME")
        password = self._get_secret_or_env("V3IO_PASSWORD")
        if self.endpoint.startswith("https://"):
            self.endpoint = self.endpoint[len("https://") :]
            self.secure = True
        elif self.endpoint.startswith("http://"):
            self.endpoint = self.endpoint[len("http://") :]
            self.secure = False
        self.client = v3io.dataplane.Client(access_key=token, endpoint=self.url)
        self.object = self.client.object
        self.auth = None
        self.token = token
        if token:
            self.headers = {"X-v3io-session-key": token}
        elif username and password:
            self.headers = basic_auth_header(username, password)

    @staticmethod
    def _do_object_request(function: callable, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except HttpResponseError as http_response_error:
            raise mlrun.errors.err_for_status_code(
                status_code=http_response_error.status_code,
                message=mlrun.errors.err_to_str(http_response_error),
            )

    @staticmethod
    def uri_to_ipython(endpoint, subpath):
        return V3IO_LOCAL_ROOT + subpath

    @property
    def url(self):
        schema = "https" if self.secure else "http"
        return f"{schema}://{self.endpoint}"

    @property
    def spark_url(self):
        return "v3io:/"

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        self._filesystem = fsspec.filesystem("v3io", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        res = dict(
            v3io_access_key=self._get_secret_or_env("V3IO_ACCESS_KEY"),
            v3io_api=mlrun.mlconf.v3io_api,
        )
        return self._sanitize_storage_options(res)

    def _upload(
        self,
        key: str,
        src_path: str,
        max_chunk_size: int = V3IO_DEFAULT_UPLOAD_CHUNK_SIZE,
    ):
        """helper function for upload method, allows for controlling max_chunk_size in testing"""
        container, path = split_path(self._join(key))
        with open(src_path, "rb") as file_obj:
            append = False
            while True:
                data = memoryview(file_obj.read(max_chunk_size))
                if not data:
                    break
                self._do_object_request(
                    self.object.put,
                    container=container,
                    path=path,
                    body=data,
                    append=append,
                )
                append = True

    def upload(self, key, src_path):
        return self._upload(key, src_path)

    def get(self, key, size=None, offset=0):
        container, path = split_path(self._join(key))
        return self._do_object_request(
            function=self.object.get,
            container=container,
            path=path,
            offset=offset,
            num_bytes=size,
        ).body

    def _put(
        self,
        key,
        data,
        append=False,
        max_chunk_size: int = V3IO_DEFAULT_UPLOAD_CHUNK_SIZE,
    ):
        """helper function for put method, allows for controlling max_chunk_size in testing"""
        data, _ = self._prepare_put_data(data, append)
        container, path = split_path(self._join(key))
        buffer_size = len(data)  # in bytes
        buffer_offset = 0
        try:
            data = memoryview(data)
        except TypeError:
            pass

        while buffer_offset < buffer_size:
            chunk_size = min(buffer_size - buffer_offset, max_chunk_size)
            append = True if buffer_offset or append else False
            self._do_object_request(
                self.object.put,
                container=container,
                path=path,
                body=data[buffer_offset : buffer_offset + chunk_size],
                append=append,
            )
            buffer_offset += chunk_size

    def put(self, key, data, append=False):
        return self._put(key, data, append)

    def stat(self, key):
        container, path = split_path(self._join(key))
        response = self._do_object_request(
            function=self.object.head, container=container, path=path
        )
        head = dict(response.headers)
        size = int(head.get("Content-Length", "0"))
        datestr = head.get("Last-Modified", "0")
        modified = time.mktime(
            datetime.strptime(datestr, "%a, %d %b %Y %H:%M:%S %Z").timetuple()
        )
        return FileStats(size, modified)

    def listdir(self, key):
        container, subpath = split_path(self._join(key))
        if not subpath.endswith("/"):
            subpath += "/"

        # without the trailing slash
        subpath_length = len(subpath) - 1

        try:
            response = self.client.container.list(
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

        file_system = self.filesystem
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
