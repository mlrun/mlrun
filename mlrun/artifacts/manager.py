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
import typing
from os.path import isdir

import mlrun.config

from ..db import RunDBInterface
from ..utils import is_legacy_artifact, is_relative_path, logger, validate_tag_name
from .base import (
    Artifact,
    DirArtifact,
    LegacyArtifact,
    LegacyDirArtifact,
    LegacyLinkArtifact,
    LinkArtifact,
)
from .dataset import (
    DatasetArtifact,
    LegacyDatasetArtifact,
    LegacyTableArtifact,
    TableArtifact,
)
from .model import LegacyModelArtifact, ModelArtifact
from .plots import (
    BokehArtifact,
    ChartArtifact,
    LegacyBokehArtifact,
    LegacyChartArtifact,
    LegacyPlotArtifact,
    LegacyPlotlyArtifact,
    PlotArtifact,
    PlotlyArtifact,
)

artifact_types = {
    "": Artifact,
    "artifact": Artifact,
    "dir": DirArtifact,
    "link": LinkArtifact,
    "plot": PlotArtifact,
    "chart": ChartArtifact,
    "table": TableArtifact,
    "model": ModelArtifact,
    "dataset": DatasetArtifact,
    "plotly": PlotlyArtifact,
    "bokeh": BokehArtifact,
}

# TODO - Remove this when legacy types are deleted in 1.5.0
legacy_artifact_types = {
    "": LegacyArtifact,
    "dir": LegacyDirArtifact,
    "link": LegacyLinkArtifact,
    "plot": LegacyPlotArtifact,
    "chart": LegacyChartArtifact,
    "table": LegacyTableArtifact,
    "model": LegacyModelArtifact,
    "dataset": LegacyDatasetArtifact,
    "plotly": LegacyPlotlyArtifact,
    "bokeh": LegacyBokehArtifact,
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

    def get_meta(self) -> dict:
        return {"kind": self.kind, "name": self.name, "tag": self.tag}


def dict_to_artifact(struct: dict) -> Artifact:
    # Need to distinguish between LegacyArtifact classes and Artifact classes. Use existence of the "metadata"
    # property to make this distinction
    kind = struct.get("kind", "")

    if is_legacy_artifact(struct):
        artifact_class = legacy_artifact_types[kind]
    else:
        artifact_class = artifact_types[kind]

    return artifact_class.from_dict(struct)


class ArtifactManager:
    def __init__(
        self,
        db: RunDBInterface = None,
        calc_hash=True,
    ):
        self.calc_hash = calc_hash

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
        producer: typing.Union["ArtifactProducer", "mlrun.MLClientCtx"],
        item: Artifact,
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
        **kwargs,
    ) -> Artifact:
        """
        Log an artifact to the DB and upload it to the artifact store.
        :param producer: The producer of the artifact, the producer depends from where the artifact is being logged.
        :param item: The artifact to log.
        :param body: The body of the artifact.
        :param target_path: The target path of the artifact. (cannot be a relative path)
                            If not provided, the artifact will be stored in the default artifact path.
                            If provided and is a remote path (e.g. s3://bucket/path), no artifact will be uploaded
                            as it already exists.
        :param tag: The tag of the artifact.
        :param viewer: Kubeflow viewer type
        :param local_path: The local path of the artifact. If remote path is provided then the artifact won't be
                            uploaded and this parameter will be used as the target path.
        :param artifact_path: The path to store the artifact.
         If not provided, the artifact will be stored in the default artifact path.
        :param format: The format of the artifact. (e.g. csv, json, html, etc.)
        :param upload: Whether to upload the artifact or not.
        :param labels: Labels to add to the artifact.
        :param db_key: The key to use when logging the artifact to the DB.
        If not provided, will generate a key based on the producer name and the artifact key.
        :param kwargs: Arguments to pass to the artifact class.
        :return: The logged artifact.
        """
        if isinstance(item, str):
            key = item
            if local_path and isdir(local_path):
                item = DirArtifact(key, body, **kwargs)
            else:
                item = Artifact(key, body, **kwargs)
        else:
            key = item.key
            target_path = target_path or item.target_path

        src_path = local_path or item.src_path  # TODO: remove src_path
        if format == "html" or (src_path and pathlib.Path(src_path).suffix == "html"):
            viewer = "web-app"
        item.format = format or item.format
        item.src_path = src_path

        if db_key is None:
            # set the default artifact db key
            if producer.kind == "run":
                db_key = producer.name + "_" + key
            else:
                db_key = key
        item.db_key = db_key if db_key else ""
        item.viewer = viewer or item.viewer
        item.tree = producer.tag
        item.tag = tag or item.tag

        item.producer = producer.get_meta()
        item.labels = labels or item.labels
        # if running as part of a workflow, enrich artifact with workflow uid label
        if item.producer.get("workflow"):
            item.labels.update({"workflow-id": item.producer.get("workflow")})

        item.iter = producer.iteration
        item.project = producer.project

        # if target_path is provided and not relative, then no need to upload the artifact as it already exists
        if target_path:
            if is_relative_path(target_path):
                raise ValueError(
                    f"target_path ({target_path}) param cannot be relative"
                )
            upload = False

        # if target_path wasn't provided, but src_path is not relative, then no need to upload the artifact as it
        # already exists. In this case set the target_path to the src_path and set upload to False
        elif src_path and "://" in src_path:
            if upload:
                raise ValueError(f"Cannot upload from remote path {src_path}")
            target_path = src_path
            upload = False

        # if mlrun.mlconf.generate_target_path_from_artifact_hash outputs True and the user
        # didn't pass target_path explicitly then we won't use `generate_target_path` to calculate the target path,
        # but rather use the `resolve_<body/file>_target_hash_path` in the `item.upload` method.
        elif (
            # if the user didn't pass target_path explicitly but asked for upload (or didn't set upload at all)
            # and the other conditions match we will enrich the target_path.
            # generally we don't want to enrich target_path if there is no matching artifact in target_path either
            # there is already a remote artifact in target_path ( that was previously uploaded ) or the user asked
            # to upload the artifact.
            (upload or upload is None)
            and not item.is_inline()
            and not mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash
        ):
            target_path = item.generate_target_path(artifact_path, producer)

        if target_path and item.is_dir and not target_path.endswith("/"):
            target_path += "/"

        item.target_path = target_path

        item.before_log()
        self.artifacts[key] = item

        if ((upload is None and item.kind != "dir") or upload) and not item.is_inline():
            # before uploading the item, we want to ensure that its tags are valid,
            # so that we don't upload something that won't be stored later
            validate_tag_name(item.metadata.tag, "artifact.metadata.tag")
            if is_legacy_artifact(item):
                item.upload()
            else:
                item.upload(artifact_path=artifact_path)

        if db_key:
            self._log_to_db(db_key, producer.project, producer.inputs, item)
        size = str(item.size) or "?"
        db_str = "Y" if (self.artifact_db and db_key) else "N"
        logger.debug(
            f"log artifact {key} at {item.target_path}, size: {size}, db: {db_str}"
        )
        return item

    def update_artifact(self, producer, item):
        self.artifacts[item.key] = item
        self._log_to_db(item.db_key, producer.project, producer.inputs, item)

    def _log_to_db(self, key, project, sources, item, tag=None):
        """
        log artifact to db
        :param key: Identifying key of the artifact.
        :param project: Project that the artifact belongs to.
        :param sources: List of artifact sources ( Mainly passed from the producer.items ).
        :param item: The actual artifact to store.
        :param tag: The name of the Tag of the artifact.
        """
        if self.artifact_db:
            item.updated = None
            if sources:
                item.sources = [{"name": k, "path": str(v)} for k, v in sources.items()]
            self.artifact_db.store_artifact(
                key,
                item.to_dict(),
                item.tree,
                iter=item.iter,
                tag=tag or item.tag,
                project=project,
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
        db_key=None,
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
            item.db_key = db_key or (name + "_" + key)
            self.artifact_db.store_artifact(
                item.db_key,
                item.to_dict(),
                item.tree,
                iter=iter,
                tag=tag,
                project=project,
            )


def extend_artifact_path(artifact_path: str, default_artifact_path: str):
    artifact_path = str(artifact_path or "")
    if artifact_path and artifact_path.startswith("+/"):
        if not default_artifact_path:
            return artifact_path[len("+/") :]
        if not default_artifact_path.endswith("/"):
            default_artifact_path += "/"
        return default_artifact_path + artifact_path[len("+/") :]
    return artifact_path or default_artifact_path


def filename(key, format):
    if not format:
        return key
    return f"{key}.{format}"
