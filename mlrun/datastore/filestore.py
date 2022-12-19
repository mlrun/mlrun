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
import time
from os import listdir, makedirs, path, stat
from shutil import copyfile

import fsspec

import mlrun

from .base import DataStore, FileStats


class FileStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, "file", endpoint, secrets=secrets)

        self._item_path, self._real_path = None, None
        if mlrun.mlconf.storage.item_to_real_path:
            # map item path prefix with real (local) path prefix e.g.:
            # item_to_real_path="\data::c:\\mlrun_data" replace \data\x.csv with c:\\mlrun_data\x.csv
            split = mlrun.mlconf.storage.item_to_real_path.split("::")
            self._item_path = split[0].strip()
            self._real_path = split[1].strip().rstrip("/").rstrip("\\")

    @property
    def url(self):
        return self.subpath

    def _join(self, key: str):
        if self._item_path and not self.subpath:
            if key.startswith(self._item_path):
                suffix = key[len(self._item_path) :]
                if suffix[0] in ["/", "\\"]:
                    suffix = suffix[1:]
                key = path.join(self._real_path, suffix)
        return path.join(self.subpath, key)

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if not self._filesystem:
            self._filesystem = fsspec.filesystem("file")
        return self._filesystem

    def get(self, key, size=None, offset=0):
        with open(self._join(key), "rb") as fp:
            if offset:
                fp.seek(offset)
            if not size:
                size = -1
            return fp.read(size)

    def put(self, key, data, append=False):
        dir_to_create = path.dirname(self._join(key))
        if dir_to_create:
            self._ensure_directory(dir_to_create)
        mode = "a" if append else "w"
        if isinstance(data, bytes):
            mode = mode + "b"
        with open(self._join(key), mode) as fp:
            fp.write(data)
            fp.close()

    def download(self, key, target_path):
        fullpath = self._join(key)
        if fullpath == target_path:
            return
        copyfile(fullpath, target_path)

    def upload(self, key, src_path):
        fullpath = self._join(key)
        if path.realpath(src_path) == path.realpath(fullpath):
            return
        dir = path.dirname(fullpath)
        if dir:
            makedirs(dir, exist_ok=True)
        copyfile(src_path, fullpath)

    def stat(self, key):
        s = stat(self._join(key))
        return FileStats(size=s.st_size, modified=s.st_mtime)

    def listdir(self, key):
        return listdir(self._join(key))

    def _ensure_directory(self, dir_to_create):
        # We retry the makedirs because it can fail if another process is creating the same dir
        # Note - inside it try to catch FileExistsError, but it still fails sometimes during its internal logic
        # where it calls `mkdir()`.
        while True:
            try:
                makedirs(dir_to_create, exist_ok=True)
                return
            except FileExistsError:
                time.sleep(0.1)
                pass
