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
import inspect
import os
import hashlib
from ..model import ModelObj
from ..datastore import StoreManager
from ..utils import DB_SCHEMA

calc_hash = True


class Artifact(ModelObj):

    _dict_fields = [
        'key', 'kind', 'iter', 'tree', 'src_path', 'target_path', 'hash',
        'description', 'viewer', 'inline', 'format', 'size', 'db_key']
    kind = ''

    def __init__(self, key=None, body=None, viewer=None, is_inline=False,
                 format=None, size=None, target_path=None):
        self.key = key
        self.project = ''
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

    def before_log(self):
        pass

    @property
    def is_dir(self):
        return False

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

    def get_store_url(self, with_tag=True, project=None):
        url = '{}://{}/{}'.format(DB_SCHEMA, project or self.project, self.db_key)
        if with_tag:
            url += '#' + self.tree

    def base_dict(self):
        return super().to_dict()

    def to_dict(self, fields=None):
        return super().to_dict(
            self._dict_fields + [
                'updated', 'labels', 'annotations', 'producer', 'sources', 'project'])

    @classmethod
    def from_dict(cls, struct=None, fields=None):
        fields = fields or cls._dict_fields + [
                'updated', 'labels', 'annotations', 'producer', 'sources', 'project']
        return super().from_dict(struct, fields=fields)

    def upload(self, data_stores: StoreManager):
        src_path = self.src_path
        body = self.get_body()
        if body:
            self._upload_body(body, data_stores)
        else:
            if src_path and os.path.isfile(src_path):
                self._upload_file(src_path, data_stores)

    def _upload_body(self, body, data_stores: StoreManager, target=None):
        if calc_hash:
            self.hash = blob_hash(body)
        self.size = len(body)
        data_stores.object(url=target or self.target_path).put(body)

    def _upload_file(self, src, data_stores: StoreManager, target=None):
        self._set_meta(src)
        data_stores.object(url=target or self.target_path).upload(src)

    def _set_meta(self, src):
        if calc_hash:
            self.hash = file_hash(src)
        self.size = os.stat(src).st_size


class DirArtifact(Artifact):
    _dict_fields = [
        'key', 'kind', 'iter', 'tree', 'src_path', 'target_path',
        'description', 'db_key']
    kind = 'dir'

    @property
    def is_dir(self):
        return True

    def upload(self, data_stores):
        if not self.src_path:
            raise ValueError('local/source path not specified')

        files = os.listdir(self.src_path)
        for f in files:
            file_path = os.path.join(self.src_path, f)
            if not os.path.isfile(file_path):
                raise ValueError('file {} not found, cant upload'.format(file_path))
            target = os.path.join(self.target_path, f)
            data_stores.object(url=target).upload(file_path)


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



