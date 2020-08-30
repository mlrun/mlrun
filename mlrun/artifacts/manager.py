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

import pathlib
from os.path import isdir

from ..datastore import StoreManager
from ..db import RunDBInterface
from ..utils import uxjoin, logger

from .base import Artifact, LinkArtifact, DirArtifact
from .plots import PlotArtifact, ChartArtifact
from .dataset import TableArtifact, DatasetArtifact
from .model import ModelArtifact

artifact_types = {
    "": Artifact,
    "dir": DirArtifact,
    "link": LinkArtifact,
    "plot": PlotArtifact,
    "chart": ChartArtifact,
    "table": TableArtifact,
    "model": ModelArtifact,
    "dataset": DatasetArtifact,
}


class ArtifactProducer:
    def __init__(self, kind, project, name, tag=None, owner=None):
        self.kind = kind
        self.project = project
        self.name = name
        self.tag = tag
        self.owner = owner
        self.uri = "/"
        self.iteration = 0
        self.inputs = {}

    def get_meta(self):
        return {"kind": self.kind, "name": self.name, "tag": self.tag}


def dict_to_artifact(struct: dict):
    kind = struct.get("kind", "")
    artifact_class = artifact_types[kind]
    return artifact_class.from_dict(struct)


class ArtifactManager:
    def __init__(
        self, stores: StoreManager, db: RunDBInterface = None, calc_hash=True,
    ):
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
        self,
        producer,
        item,
        body=None,
        target_path="",
        tag="",
        viewer="",
        local_path="",
        artifact_path=None,
        format=None,
        upload=None,
        labels=None,
        db_key=None,
    ):
        if isinstance(item, str):
            key = item
            if local_path and isdir(local_path):
                item = DirArtifact(key, body)
            else:
                item = Artifact(key, body)
        else:
            key = item.key
            target_path = target_path or item.target_path

        src_path = local_path or item.src_path  # TODO: remove src_path
        if format == "html" or (src_path and pathlib.Path(src_path).suffix == "html"):
            viewer = "web-app"
        item.format = format or item.format
        item.src_path = src_path
        if src_path and ("://" in src_path or src_path.startswith("/")):
            raise ValueError(
                "local/source path ({}) must be a relative path, "
                "cannot be remote or absolute path, "
                "use target_path for absolute paths".format(src_path)
            )

        if target_path:
            if not (target_path.startswith("/") or "://" in target_path):
                raise ValueError(
                    "target_path ({}) param cannot be relative".format(target_path)
                )
        else:
            target_path = uxjoin(
                artifact_path,
                src_path,
                filename(key, item.format),
                producer.iteration,
                item.is_dir,
            )

        if item.is_dir and not target_path.endswith("/"):
            target_path += "/"

        item.target_path = target_path
        item.viewer = viewer or item.viewer
        item.tree = producer.tag
        item.labels = labels or item.labels
        item.producer = producer.get_meta()
        item.iter = producer.iteration
        item.project = producer.project

        if db_key is None:
            # set the default artifact db key
            if producer.kind == "run":
                db_key = producer.name + "_" + key
            else:
                db_key = key
        item.db_key = db_key if db_key else ""

        item.before_log()
        self.artifacts[key] = item

        if (upload is None and item.kind != "dir") or upload:
            item.upload(self.data_stores)

        if db_key:
            self._log_to_db(db_key, producer.project, producer.inputs, item, tag)
        size = str(item.size) or "?"
        logger.debug(
            "log artifact {} at {}, size: {}, db: {}".format(
                key,
                item.target_path,
                size,
                "Y" if (self.artifact_db and db_key) else "N",
            )
        )
        return item

    def _log_to_db(self, key, project, sources, item, tag):
        if self.artifact_db:
            if sources:
                item.sources = [{"name": k, "path": str(v)} for k, v in sources.items()]
            self.artifact_db.store_artifact(
                key, item.to_dict(), item.tree, iter=item.iter, tag=tag, project=project
            )

    def link_artifact(
        self,
        project,
        name,
        tree,
        key,
        iter=0,
        artifact_path="",
        tag="",
        link_iteration=0,
        link_key=None,
        link_tree=None,
    ):
        if self.artifact_db:
            item = LinkArtifact(
                key,
                artifact_path,
                link_iteration=link_iteration,
                link_key=link_key,
                link_tree=link_tree,
            )
            item.tree = tree
            item.iter = iter
            item.db_key = name + "_" + key
            self.artifact_db.store_artifact(
                item.db_key,
                item.to_dict(),
                item.tree,
                iter=iter,
                tag=tag,
                project=project,
            )

    def _get_store(self, url):
        return self.data_stores.get_or_create_store(url)


def filename(key, format):
    if not format:
        return key
    return "{}.{}".format(key, format)
