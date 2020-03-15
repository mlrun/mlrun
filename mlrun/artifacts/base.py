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

from ..model import ModelObj


class Artifact(ModelObj):

    _dict_fields = [
        'key', 'kind', 'iter', 'tree', 'src_path', 'target_path', 'hash',
        'description', 'viewer', 'inline', 'format', 'size']
    kind = ''

    def __init__(self, key, body=None, src_path=None, target_path='',
                 viewer=None, inline=False, format=None, size=None):
        self.key = key
        self.size = size
        self.iter = None
        self.tree = None
        self.updated = None
        self.target_path = target_path
        self.src_path = src_path
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
        self._inline = inline
        self.license = ''
        self._post_init()

    def _post_init(self):
        pass

    @property
    def inline(self):
        if self._inline:
            return self.get_body()
        return None

    def get_body(self):
        return self._body

    def base_dict(self):
        return super().to_dict()

    def to_dict(self, fields=None):
        return super().to_dict(
            self._dict_fields + [
                'updated', 'labels', 'annotations', 'producer', 'sources'])


class LinkArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ['link_iteration', 'link_key', 'link_tree']
    kind = 'link'

    def __init__(self, key, target_path='', link_iteration=None,
                 link_key=None, link_tree=None):

        super().__init__(key, target_path=target_path)
        self.link_iteration = link_iteration
        self.link_key = link_key
        self.link_tree = link_tree


