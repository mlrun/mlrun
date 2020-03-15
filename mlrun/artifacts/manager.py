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
import pathlib

from ..datastore import StoreManager
from ..db import RunDBInterface
from ..utils import uxjoin, run_keys, logger

from .base import Artifact, LinkArtifact
from .plots import PlotArtifact, ChartArtifact
from .dataset import TableArtifact
from .model import ModelArtifact

artifact_types = {
    '': Artifact,
    'link': LinkArtifact,
    'plot': PlotArtifact,
    'chart': ChartArtifact,
    'table': TableArtifact,
    'model': ModelArtifact,
}


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


class ArtifactManager:

    def __init__(self, stores: StoreManager,
                 db: RunDBInterface = None,
                 out_path='',
                 calc_hash=True):
        self.out_path = out_path
        self.calc_hash = calc_hash

        self.data_stores = stores
        self.artifact_db = db
        self.input_artifacts = {}
        self.artifacts = {}

    def artifact_list(self, full=False):
        artifacts = []
        for artifact in self.artifacts.values():
            if isinstance(artifact, dict):
                artifacts.append(artifact)
            else:
                if full:
                    artifacts.append(artifact.to_dict())
                else:
                    artifacts.append(artifact.base_dict())
        return artifacts

    def log_artifact(
        self, execution, item, body=None, target_path='', src_path='', tag='',
            viewer='', local_path='', artifact_path=None, format=None,
            upload=True, labels=None):
        if isinstance(item, str):
            key = item
            item = Artifact(key, body)
        else:
            key = item.key
            target_path = target_path or item.target_path

        src_path = src_path or local_path or item.src_path  # TODO: remove src_path
        if format == 'html' or (
                src_path and pathlib.Path(src_path).suffix == 'html'):
            viewer = 'web-app'
        item.format = format or item.format
        item.src_path = src_path
        if item.src_path and '://' in item.src_path:
            raise ValueError('source path cannot be a remote URL')

        artifact_path = artifact_path or self.out_path
        if artifact_path or not target_path:
            target_path = uxjoin(
                artifact_path, src_path or filename(key, item.format),
                execution.iteration)
        elif not (target_path.startswith('/') or '://' in target_path):
            target_path = uxjoin(
                self.out_path, target_path, execution.iteration)
        item.target_path = target_path
        item.viewer = viewer or item.viewer

        item.tree = execution.tag
        if labels:
            if not item.labels:
                item.labels = {}
            for k, v in labels.items():
                item.labels[k] = str(v)

        self.artifacts[key] = item

        if upload:
            store, ipath = self._get_store(target_path)
            body = item.get_body()
            if body:
                if self.calc_hash:
                    item.hash = blob_hash(body)
                item.size = len(body)
                store.put(ipath, body)
            else:
                if src_path and os.path.isfile(src_path):
                    if self.calc_hash:
                        item.hash = file_hash(src_path)
                    item.size = os.stat(src_path).st_size
                    store.upload(ipath, src_path)

        if self.artifact_db:
            if not item.sources:
                sources = execution.to_dict()['spec'][run_keys.inputs]
                if sources:
                    item.sources = [{'name': k, 'path': v}
                                    for k, v in sources.items()]
            item.producer = execution.get_meta()
            item.iter = execution.iteration
            self.artifact_db.store_artifact(key, item.to_dict(), item.tree,
                                            iter=execution.iteration, tag=tag,
                                            project=execution.project)

        size = str(item.size) or '?'
        logger.info('log artifact {} at {}, size: {}, db: {}'.format(
            key, target_path, size, 'Y' if self.artifact_db else 'N'
        ))

    def link_artifact(self, execution, key, artifact_path='', tag='',
                      link_iteration=0, link_key=None, link_tree=None):
        if self.artifact_db:
            item = LinkArtifact(key, artifact_path,
                                link_iteration=link_iteration,
                                link_key=link_key,
                                link_tree=link_tree)
            item.tree = execution.tag
            item.iter = execution.iteration
            self.artifact_db.store_artifact(key, item.to_dict(), item.tree,
                                            iter=execution.iteration, tag=tag,
                                            project=execution.project)

    def _get_store(self, url):
        return self.data_stores.get_or_create_store(url)


def filename(key, format):
    if not format:
        return key
    return '{}.{}'.format(key, format)