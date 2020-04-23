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

from os import path, makedirs, listdir, stat
from shutil import copyfile

from .base import DataStore, FileStats

class FileStore(DataStore):
    def __init__(self, parent, schema, name, endpoint=''):
        super().__init__(parent, name, 'file', endpoint)

    @property
    def url(self):
        return self.subpath

    def _join(self, key):
        return path.join(self.subpath, key)

    def get(self, key, size=None, offset=0):
        with open(self._join(key), 'rb') as fp:
            if offset:
                fp.seek(offset)
            if not size:
                size = -1
            return fp.read(size)

    def put(self, key, data, append=False):
        dir = path.dirname(self._join(key))
        if dir:
            makedirs(dir, exist_ok=True)
        mode = 'a' if append else 'w'
        if isinstance(data, bytes):
            mode = mode + 'b'
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
        if fullpath == src_path:
            return
        dir = path.dirname(fullpath)
        if dir:
            makedirs(dir, exist_ok=True)
        copyfile(src_path, fullpath)

    def stat(self, key):
        s = stat(self._join(key))
        return FileStats(size=s.st_size, modified=s.st_mtime)

    def listdir(self, key):
        return listdir(key)
