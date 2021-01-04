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

import yaml

import mlrun
from ..model import ModelObj
from ..datastore import is_store_uri, get_store_uri, store_manager
from ..utils import StorePrefix

calc_hash = True


class Artifact(ModelObj):

    _dict_fields = [
        "key",
        "kind",
        "iter",
        "tree",
        "src_path",
        "target_path",
        "hash",
        "description",
        "viewer",
        "inline",
        "format",
        "size",
        "db_key",
        "extra_data",
    ]
    kind = ""

    def __init__(
        self,
        key=None,
        body=None,
        viewer=None,
        is_inline=False,
        format=None,
        size=None,
        target_path=None,
    ):
        self.key = key
        self.project = ""
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
        self.license = ""
        self.extra_data = {}

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

    def get_target_path(self):
        return self.target_path

    def get_store_url(self, with_tag=True, project=None):
        uri = "{}/{}".format(project or self.project, self.db_key)
        if with_tag:
            uri += "#" + self.tree
        return get_store_uri(StorePrefix.Artifact, uri)

    def base_dict(self):
        return super().to_dict()

    def to_dict(self, fields=None):
        return super().to_dict(
            self._dict_fields
            + ["updated", "labels", "annotations", "producer", "sources", "project"]
        )

    @classmethod
    def from_dict(cls, struct=None, fields=None):
        fields = fields or cls._dict_fields + [
            "updated",
            "labels",
            "annotations",
            "producer",
            "sources",
            "project",
        ]
        return super().from_dict(struct, fields=fields)

    def upload(self):
        src_path = self.src_path
        body = self.get_body()
        if body:
            self._upload_body(body)
        else:
            if src_path and os.path.isfile(src_path):
                self._upload_file(src_path)

    def _upload_body(self, body, target=None):
        if calc_hash:
            self.hash = blob_hash(body)
        self.size = len(body)
        store_manager.object(url=target or self.target_path).put(body)

    def _upload_file(self, src, target=None):
        self._set_meta(src)
        store_manager.object(url=target or self.target_path).upload(src)

    def _set_meta(self, src):
        if calc_hash:
            self.hash = file_hash(src)
        self.size = os.stat(src).st_size


class DirArtifact(Artifact):
    _dict_fields = [
        "key",
        "kind",
        "iter",
        "tree",
        "src_path",
        "target_path",
        "description",
        "db_key",
    ]
    kind = "dir"

    @property
    def is_dir(self):
        return True

    def upload(self):
        if not self.src_path:
            raise ValueError("local/source path not specified")

        files = os.listdir(self.src_path)
        for f in files:
            file_path = os.path.join(self.src_path, f)
            if not os.path.isfile(file_path):
                raise ValueError("file {} not found, cant upload".format(file_path))
            target = os.path.join(self.target_path, f)
            store_manager.object(url=target).upload(file_path)


class LinkArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ["link_iteration", "link_key", "link_tree"]
    kind = "link"

    def __init__(
        self,
        key=None,
        target_path="",
        link_iteration=None,
        link_key=None,
        link_tree=None,
    ):

        super().__init__(key)
        self.target_path = target_path
        self.link_iteration = link_iteration
        self.link_key = link_key
        self.link_tree = link_tree


def file_hash(filename):
    h = hashlib.sha1()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def blob_hash(data):
    if isinstance(data, str):
        data = data.encode()
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def upload_extra_data(
    artifact_spec: Artifact, extra_data: dict, prefix="", update_spec=False,
):
    if not extra_data:
        return
    target_path = artifact_spec.target_path
    for key, item in extra_data.items():

        if isinstance(item, bytes):
            target = os.path.join(target_path, key)
            store_manager.object(url=target).put(item)
            artifact_spec.extra_data[prefix + key] = target
            continue

        if not (item.startswith("/") or "://" in item):
            src_path = (
                os.path.join(artifact_spec.src_path, item)
                if artifact_spec.src_path
                else item
            )
            if not os.path.isfile(src_path):
                raise ValueError("extra data file {} not found".format(src_path))
            target = os.path.join(target_path, item)
            store_manager.object(url=target).upload(src_path)

        if update_spec:
            artifact_spec.extra_data[prefix + key] = item


def get_artifact_meta(artifact):
    """return artifact object, and list of extra data items


    :param artifact:   artifact path (store://..) or DataItem

    :returns: artifact object, extra data dict

    """
    if hasattr(artifact, "artifact_url"):
        artifact = artifact.artifact_url

    if is_store_uri(artifact):
        artifact_spec, target = store_manager.get_store_artifact(artifact)

    elif artifact.lower().endswith(".yaml"):
        data = store_manager.object(url=artifact).get()
        spec = yaml.load(data, Loader=yaml.FullLoader)
        artifact_spec = mlrun.artifacts.dict_to_artifact(spec)

    else:
        raise ValueError("cant resolve artifact file for {}".format(artifact))

    extra_dataitems = {}
    for k, v in artifact_spec.extra_data.items():
        extra_dataitems[k] = store_manager.object(v, key=k)

    return artifact_spec, extra_dataitems
