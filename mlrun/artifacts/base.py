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

import os
import hashlib
from ..model import ModelObj

calc_hash = True


class Artifact(ModelObj):

    _dict_fields = [
        'key', 'kind', 'iter', 'tree', 'src_path', 'target_path', 'hash',
        'description', 'viewer', 'inline', 'format', 'size', 'db_key']
    kind = ''

    def __init__(self, key=None, body=None, viewer=None, is_inline=False,
                 format=None, size=None, target_path=None):
        self.key = key
        self.db_key = None
        self.size = size
        self.iter = None
        self.tree = None
        self.updated = None
        self.target_path = target_path
        self.src_path = None
        self._body = body
        self.format = format
        self.description = None
        self.viewer = viewer
        self.encoding = None
        self.labels = None
        self.annotations = None
        self.sources = []
        self.producer = None
        self.hash = None
        self._inline = is_inline
        self.license = ''
        self._post_init()

    def _post_init(self):
        pass

    @property
    def inline(self):
        if self._inline:
            return self.get_body()
        return None

    @inline.setter
    def inline(self, body):
        self._body = body

    def get_body(self):
        return self._body

    def base_dict(self):
        return super().to_dict()

    def to_dict(self, fields=None):
        return super().to_dict(
            self._dict_fields + [
                'updated', 'labels', 'annotations', 'producer', 'sources'])

    def upload(self, data_stores):
        src_path = self.src_path
        store, ipath = data_stores.get_or_create_store(self.target_path)
        body = self.get_body()
        if body:
            if calc_hash:
                self.hash = blob_hash(body)
            self.size = len(body)
            store.put(ipath, body)
        else:
            if src_path and os.path.isfile(src_path):
                self._upload_file(src_path, data_stores)

    def _upload_file(self, src, data_stores):
        store, ipath = data_stores.get_or_create_store(self.target_path)
        self._set_meta(src)
        store.upload(ipath, src)

    def _set_meta(self, src):
        if calc_hash:
            self.hash = file_hash(src)
        self.size = os.stat(src).st_size



class LinkArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['link_iteration', 'link_key', 'link_tree']
    kind = 'link'

    def __init__(self, key=None, target_path='', link_iteration=None,
                 link_key=None, link_tree=None):

        super().__init__(key)
        self.target_path = target_path
        self.link_iteration = link_iteration
        self.link_key = link_key
        self.link_tree = link_tree


def file_hash(filename):
    h = hashlib.sha1()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def blob_hash(data):
    if isinstance(data, str):
        data = data.encode()
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()



