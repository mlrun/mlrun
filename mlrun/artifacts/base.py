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
import hashlib
import os
import pathlib
import tempfile
import typing
import warnings
import zipfile

import yaml

import mlrun
import mlrun.artifacts
import mlrun.datastore
import mlrun.errors

from ..model import ModelObj
from ..utils import (
    StorePrefix,
    calculate_local_file_hash,
    generate_artifact_uri,
    is_relative_path,
)


class ArtifactMetadata(ModelObj):
    _dict_fields = [
        "key",
        "project",
        "iter",
        "tree",
        "description",
        "hash",
        "tag",
        "uid",
    ]
    _extra_fields = ["updated", "labels"]

    def __init__(
        self,
        key=None,
        project=None,
        iter=None,
        tree=None,
        description=None,
        hash=None,
        tag=None,
        uid=None,
    ):
        self.key = key
        self.project = project
        self.iter = iter
        self.tree = tree
        self.description = description
        self.hash = hash
        self.labels = {}
        self.updated = None
        self.tag = tag  # temp store of the tag
        self.uid = uid

    def base_dict(self):
        return super().to_dict()

    def to_dict(
        self,
        fields: typing.Optional[list] = None,
        exclude: typing.Optional[list] = None,
        strip: bool = False,
    ):
        """return long dict form of the artifact"""
        return super().to_dict(
            self._dict_fields + self._extra_fields, exclude=exclude, strip=strip
        )

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: typing.Optional[dict] = None
    ):
        fields = fields or cls._dict_fields + cls._extra_fields
        return super().from_dict(
            struct, fields=fields, deprecated_fields=deprecated_fields
        )


class ArtifactSpec(ModelObj):
    _dict_fields = [
        "src_path",
        "target_path",
        "viewer",
        "inline",
        "format",
        "size",
        "db_key",
        "extra_data",
        "unpackaging_instructions",
        "producer",
    ]

    _extra_fields = ["annotations", "sources", "license", "encoding"]
    _exclude_fields_from_uid_hash = [
        # if the artifact is first created, it will not have a db_key,
        # exclude it so further updates of the artifacts will have the same hash
        "db_key",
        "extra_data",
    ]

    def __init__(
        self,
        src_path=None,
        target_path=None,
        viewer=None,
        is_inline=False,
        format=None,
        size=None,
        db_key=None,
        extra_data=None,
        body=None,
        unpackaging_instructions: typing.Optional[dict] = None,
    ):
        self.src_path = src_path
        self.target_path = target_path
        self.viewer = viewer
        self._is_inline = is_inline
        self.format = format
        self.size = size
        self.db_key = db_key
        self.extra_data = extra_data or {}
        self.unpackaging_instructions = unpackaging_instructions

        self._body = body
        self.encoding = None
        self.annotations = None
        self.sources = []
        self.producer = None
        self.license = ""

    def base_dict(self):
        return super().to_dict()

    def to_dict(
        self,
        fields: typing.Optional[list] = None,
        exclude: typing.Optional[list] = None,
        strip: bool = False,
    ):
        """return long dict form of the artifact"""
        return super().to_dict(
            self._dict_fields + self._extra_fields, exclude=exclude, strip=strip
        )

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: typing.Optional[dict] = None
    ):
        fields = fields or cls._dict_fields + cls._extra_fields
        return super().from_dict(
            struct, fields=fields, deprecated_fields=deprecated_fields
        )

    @property
    def inline(self):
        """inline data (body)"""

        if self._is_inline:
            return self.get_body()
        return None

    @inline.setter
    def inline(self, body):
        self._body = body
        if body:
            self._is_inline = True

    def get_body(self):
        """get the artifact body when inline"""
        return self._body


class ArtifactStatus(ModelObj):
    _dict_fields = ["state", "stats", "preview", "header_original_length"]

    def __init__(self):
        self.state = "created"
        self.stats = None
        self.preview = None
        self.header_original_length = None

    def base_dict(self):
        return super().to_dict()


class Artifact(ModelObj):
    kind = "artifact"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    _store_prefix = StorePrefix.Artifact

    def __init__(
        self,
        key=None,
        body=None,
        viewer=None,
        is_inline=False,
        format=None,
        size=None,
        target_path=None,
        project=None,
        src_path: typing.Optional[str] = None,
        # All params up until here are legacy params for compatibility with legacy artifacts.
        # TODO: remove them in 1.10.0.
        metadata: ArtifactMetadata = None,
        spec: ArtifactSpec = None,
    ):
        if (
            key
            or body
            or viewer
            or is_inline
            or format
            or size
            or target_path
            or project
            or src_path
        ):
            warnings.warn(
                "Artifact constructor parameters are deprecated in 1.7.0 and will be removed in 1.10.0. "
                "Use the metadata and spec parameters instead.",
                DeprecationWarning,
            )

        self._metadata = None
        self.metadata = metadata
        self._spec = None
        self.spec = spec

        self.metadata.key = key or self.metadata.key
        self.metadata.project = (
            project or mlrun.mlconf.default_project or self.metadata.project
        )
        self.spec.size = size or self.spec.size
        self.spec.target_path = target_path or self.spec.target_path
        self.spec.format = format or self.spec.format
        self.spec.viewer = viewer or self.spec.viewer
        self.spec.src_path = src_path

        if body:
            self.spec._body = body
        self.spec._is_inline = is_inline or self.spec._is_inline

        self.status = ArtifactStatus()

    @property
    def metadata(self) -> ArtifactMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", ArtifactMetadata)

    @property
    def spec(self) -> ArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ArtifactSpec)

    @property
    def status(self) -> ArtifactStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", ArtifactStatus)

    def _get_file_body(self):
        body = self.spec.get_body()
        if body:
            return body
        if self.src_path and os.path.isfile(self.src_path):
            with open(self.src_path, "rb") as fp:
                return fp.read()
        return mlrun.get_dataitem(self.get_target_path()).get()

    def export(self, target_path: str, with_extras=True):
        """save the artifact object into a yaml/json file or zip archive

        when the target path is a .yaml/.json file the artifact spec is saved into that file,
        when the target_path suffix is '.zip' the artifact spec, body and extra data items are
        packaged into a zip file. The archive target_path support DataItem urls for remote object storage
        (e.g. s3://<bucket>/<path>).

        :param target_path: path to store artifact .yaml/.json spec or .zip (spec with the content)
        :param with_extras: will include the extra_data items in the zip archive
        """
        if target_path.endswith(".yaml") or target_path.endswith(".yml"):
            mlrun.get_dataitem(target_path).put(self.to_yaml())

        elif target_path.endswith(".json"):
            mlrun.get_dataitem(target_path).put(self.to_json())

        elif target_path.endswith(".zip"):
            tmp_path = None
            if "://" in target_path:
                tmp_path = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
            zipf = zipfile.ZipFile(tmp_path or target_path, "w")
            body = self._get_file_body()
            zipf.writestr("_body", body)
            extras = {}
            if with_extras:
                for k, item_path in self.extra_data.items():
                    if is_relative_path(item_path):
                        base_dir = self.src_path or ""
                        if not self.is_dir:
                            base_dir = os.path.dirname(base_dir)
                        item_path = os.path.join(base_dir, item_path).replace("\\", "/")
                    zipf.writestr(k, mlrun.get_dataitem(item_path).get())
                    extras[k] = k
            artifact = self.copy()
            artifact.extra_data = extras
            zipf.writestr("_spec.yaml", artifact.to_yaml())
            zipf.close()

            if tmp_path:
                mlrun.get_dataitem(target_path).upload(tmp_path)
                os.remove(tmp_path)
        else:
            raise ValueError("unsupported file suffix, use .yaml, .json, or .zip")

    def before_log(self):
        for key, item in self.spec.extra_data.items():
            if hasattr(item, "get_target_path"):
                self.spec.extra_data[key] = item.get_target_path()

    @property
    def is_dir(self):
        """this is a directory"""
        return False

    @property
    def uri(self):
        """return artifact uri (store://..)"""
        return self.get_store_url()

    def to_dataitem(self):
        """return a DataItem object (if available) representing the artifact content"""
        uri = self.get_store_url()
        if uri:
            return mlrun.get_dataitem(uri)

    def get_body(self):
        """get the artifact body when inline"""
        return self.spec.get_body()

    def get_target_path(self):
        """get the absolute target path for the artifact"""
        return self.spec.target_path

    def get_store_url(self, with_tag=True, project=None, with_tree=True):
        """get the artifact uri (store://..) with optional parameters"""
        tag = self.metadata.tag if with_tag else None
        tree = self.metadata.tree if with_tree else None

        uri = generate_artifact_uri(
            project or self.metadata.project,
            self.spec.db_key,
            iter=self.metadata.iter,
            tree=tree,
            tag=tag,
            uid=self.uid,
        )
        return mlrun.datastore.get_store_uri(self._store_prefix, uri)

    def base_dict(self):
        """return short dict form of the artifact"""
        struct = {"kind": self.kind}
        for field in ["metadata", "spec", "status"]:
            val = getattr(self, field, None)
            if val:
                struct[field] = val.base_dict()
        return struct

    def upload(self, artifact_path: typing.Optional[str] = None):
        """
        internal, upload to target store
        :param artifact_path: required only for when generating target_path from artifact hash
        """
        src_path = self.spec.src_path
        body = self.get_body()
        if body:
            self._upload_body(body=body, artifact_path=artifact_path)
        else:
            if src_path and os.path.isfile(src_path):
                self._upload_file(source_path=src_path, artifact_path=artifact_path)

    def _upload_body(
        self, body, target=None, artifact_path: typing.Optional[str] = None
    ):
        body_hash = None
        if not target and not self.spec.target_path:
            if not mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Unable to resolve target path, no target path is defined and "
                    "mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash is set to false"
                )
            body_hash, self.spec.target_path = self.resolve_body_target_hash_path(
                body, artifact_path
            )

        if mlrun.mlconf.artifacts.calculate_hash:
            self.metadata.hash = body_hash or calculate_blob_hash(body)
        self.spec.size = len(body)

        mlrun.datastore.store_manager.object(url=target or self.spec.target_path).put(
            body
        )

    def _upload_file(
        self,
        source_path: str,
        target_path: typing.Optional[str] = None,
        artifact_path: typing.Optional[str] = None,
    ):
        file_hash = None
        if not target_path and not self.spec.target_path:
            if not mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Unable to resolve target path, no target path is defined and "
                    "mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash is set to false"
                )
            file_hash, self.spec.target_path = self.resolve_file_target_hash_path(
                source_path, artifact_path
            )
        if mlrun.mlconf.artifacts.calculate_hash:
            self.metadata.hash = file_hash or calculate_local_file_hash(source_path)
        self.spec.size = os.stat(source_path).st_size

        mlrun.datastore.store_manager.object(
            url=target_path or self.spec.target_path
        ).upload(source_path)

    def resolve_body_target_hash_path(
        self, body: typing.Union[bytes, str], artifact_path: str
    ) -> (str, str):
        """
        constructs the target path by calculating the artifact body hash
        :param body: artifact body to calculate hash on
        :param artifact_path: the base path for constructing the target path
        :return: [artifact_hash, target_path]
        """
        return self._resolve_target_hash_path(
            artifact_source=body,
            artifact_path=artifact_path,
            hash_method=calculate_blob_hash,
        )

    def resolve_file_target_hash_path(
        self, source_path: str, artifact_path: str
    ) -> (str, str):
        """
        constructs the target path by calculating the artifact source hash
        :param source_path: artifact file source path to calculate hash on
        :param artifact_path: the base path for constructing the target path
        :return: [artifact_hash, target_path]
        """
        return self._resolve_target_hash_path(
            artifact_source=source_path,
            artifact_path=artifact_path,
            hash_method=calculate_local_file_hash,
        )

    def _resolve_target_hash_path(
        self,
        artifact_source: typing.Union[bytes, str],
        artifact_path: str,
        hash_method: typing.Callable,
    ) -> (str, str):
        """
        constructs the target path by calculating the artifact source hash
        :param artifact_source: artifact to calculate hash on. May be path to source (str) or content (bytes)
        :param artifact_path: the base path for constructing the target path
        :param hash_method: the method which calculates the hash from the artifact source
        :return: [artifact_hash, target_path]
        """
        if not artifact_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Unable to resolve file target hash path, artifact_path is not defined"
            )
        artifact_hash = hash_method(artifact_source)
        suffix = self._resolve_suffix()
        artifact_path = (
            artifact_path + "/" if not artifact_path.endswith("/") else artifact_path
        )
        target_path = f"{artifact_path}{artifact_hash}{suffix}"
        return artifact_hash, target_path

    def _resolve_suffix(self) -> str:
        suffix = "".join(pathlib.Path(self.spec.src_path or "").suffixes)
        if not suffix and self.spec.format:
            suffix = f".{self.spec.format}"
        return suffix

    # Following properties are for backwards compatibility with the ArtifactLegacy class. They should be
    # removed once we only work with the new Artifact structure.

    def is_inline(self):
        return self.spec._is_inline

    @property
    def inline(self):
        return self.spec.inline

    @inline.setter
    def inline(self, body):
        self.spec.inline = body

    @property
    def tag(self):
        return self.metadata.tag

    @tag.setter
    def tag(self, tag):
        self.metadata.tag = tag

    @property
    def key(self):
        return self.metadata.key

    @key.setter
    def key(self, key):
        self.metadata.key = key

    @property
    def src_path(self):
        return self.spec.src_path

    @src_path.setter
    def src_path(self, src_path):
        self.spec.src_path = src_path

    @property
    def target_path(self):
        return self.spec.target_path

    @target_path.setter
    def target_path(self, target_path):
        self.spec.target_path = target_path

    @property
    def producer(self):
        return self.spec.producer

    @producer.setter
    def producer(self, producer):
        self.spec.producer = producer

    @property
    def format(self):
        return self.spec.format

    @format.setter
    def format(self, format):
        self.spec.format = format

    @property
    def viewer(self):
        return self.spec.viewer

    @viewer.setter
    def viewer(self, viewer):
        self.spec.viewer = viewer

    @property
    def size(self):
        return self.spec.size

    @size.setter
    def size(self, size):
        self.spec.size = size

    @property
    def db_key(self):
        return self.spec.db_key

    @db_key.setter
    def db_key(self, db_key):
        self.spec.db_key = db_key

    @property
    def sources(self):
        return self.spec.sources

    @sources.setter
    def sources(self, sources):
        self.spec.sources = sources

    @property
    def extra_data(self):
        return self.spec.extra_data

    @extra_data.setter
    def extra_data(self, extra_data):
        self.spec.extra_data = extra_data

    @property
    def labels(self):
        return self.metadata.labels

    @labels.setter
    def labels(self, labels):
        self.metadata.labels = labels

    @property
    def iter(self):
        return self.metadata.iter

    @iter.setter
    def iter(self, iter):
        self.metadata.iter = iter

    @property
    def tree(self):
        return self.metadata.tree

    @tree.setter
    def tree(self, tree):
        self.metadata.tree = tree

    @property
    def project(self):
        return self.metadata.project

    @project.setter
    def project(self, project):
        self.metadata.project = project

    @property
    def hash(self):
        return self.metadata.hash

    @hash.setter
    def hash(self, hash):
        self.metadata.hash = hash

    @property
    def uid(self):
        return self.metadata.uid

    @uid.setter
    def uid(self, uid):
        self.metadata.uid = uid

    def generate_target_path(self, artifact_path, producer):
        return generate_target_path(self, artifact_path, producer)


class DirArtifactSpec(ArtifactSpec):
    _dict_fields = [
        "src_path",
        "target_path",
        "db_key",
        "producer",
    ]


class DirArtifact(Artifact):
    kind = "dir"

    @property
    def spec(self) -> DirArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", DirArtifactSpec)

    @property
    def is_dir(self):
        return True

    def upload(self, artifact_path: typing.Optional[str] = None):
        """
        internal, upload to target store
        :param artifact_path: required only for when generating target_path from artifact hash
        """
        if not self.spec.src_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "local/source path not specified"
            )

        files = os.listdir(self.spec.src_path)
        for file_name in files:
            file_path = os.path.join(self.spec.src_path, file_name)
            if not os.path.isfile(file_path):
                raise mlrun.errors.MLRunNotFoundError(
                    f"file {file_path} not found, cant upload"
                )

            if self.spec.target_path:
                target_path = os.path.join(self.spec.target_path, file_name)
            elif mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash:
                _, target_path = self.resolve_file_target_hash_path(
                    source_path=file_path, artifact_path=artifact_path
                )
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "target path is not specified and mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash "
                    "set to False"
                )

            mlrun.datastore.store_manager.object(url=target_path).upload(file_path)
            # add files of the directory to the extra data of the artifact with value of the target path
            self.spec.extra_data[file_name] = target_path


class LinkArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "link_iteration",
        "link_key",
        "link_tree",
    ]

    def __init__(
        self,
        src_path=None,
        target_path=None,
        link_iteration=None,
        link_key=None,
        link_tree=None,
    ):
        super().__init__(src_path, target_path)
        self.link_iteration = link_iteration
        self.link_key = link_key
        self.link_tree = link_tree


class LinkArtifact(Artifact):
    kind = "link"

    def __init__(
        self,
        key=None,
        target_path="",
        link_iteration=None,
        link_key=None,
        link_tree=None,
        project=None,
        # All params up until here are legacy params for compatibility with legacy artifacts.
        # TODO: remove them in 1.10.0.
        metadata: ArtifactMetadata = None,
        spec: LinkArtifactSpec = None,
    ):
        if key or target_path or link_iteration or link_key or link_tree or project:
            warnings.warn(
                "Artifact constructor parameters are deprecated in 1.7.0 and will be removed in 1.10.0. "
                "Use the metadata and spec parameters instead.",
                DeprecationWarning,
            )
        super().__init__(
            key, target_path=target_path, project=project, metadata=metadata, spec=spec
        )
        self.spec.link_iteration = link_iteration
        self.spec.link_key = link_key
        self.spec.link_tree = link_tree

    @property
    def spec(self) -> LinkArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", LinkArtifactSpec)


def calculate_blob_hash(data):
    if isinstance(data, str):
        data = data.encode()
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def upload_extra_data(
    artifact: Artifact,
    extra_data: dict,
    prefix="",
    update_spec=False,
    artifact_path: typing.Optional[str] = None,
):
    """upload extra data to the artifact store"""
    if not extra_data:
        return
    target_path = artifact.target_path
    for key, item in extra_data.items():
        if item is ...:
            # Skip future links (packagers feature for linking artifacts before they are logged)
            continue

        if isinstance(item, bytes):
            if target_path:
                target = os.path.join(target_path, prefix + key)
            else:
                _, target = artifact.resolve_body_target_hash_path(
                    item, artifact_path=artifact_path
                )

            mlrun.datastore.store_manager.object(url=target).put(item)
            artifact.extra_data[prefix + key] = target
            continue

        if is_relative_path(item):
            src_path = (
                os.path.join(artifact.src_path, item) if artifact.src_path else item
            )
            if not os.path.isfile(src_path):
                raise ValueError(f"Extra data file {src_path} not found")

            if target_path:
                target = os.path.join(target_path, item)
            else:
                _, target = artifact.resolve_file_target_hash_path(
                    src_path, artifact_path=artifact_path
                )
            mlrun.datastore.store_manager.object(url=target).upload(src_path)
            artifact.extra_data[prefix + key] = target
            continue

        if update_spec:
            artifact.extra_data[prefix + key] = item


def get_artifact_meta(artifact):
    """return artifact object, and list of extra data items


    :param artifact:   artifact path (store://..) or DataItem

    :returns: artifact object, extra data dict

    """
    if hasattr(artifact, "artifact_url"):
        artifact = artifact.artifact_url

    if mlrun.datastore.is_store_uri(artifact):
        artifact_spec, target = mlrun.datastore.store_manager.get_store_artifact(
            artifact
        )

    elif artifact.lower().endswith(".yaml"):
        data = mlrun.datastore.store_manager.object(url=artifact).get()
        spec = yaml.load(data, Loader=yaml.FullLoader)
        artifact_spec = mlrun.artifacts.dict_to_artifact(spec)

    else:
        raise ValueError(f"cant resolve artifact file for {artifact}")

    extra_dataitems = {}
    for k, v in artifact_spec.extra_data.items():
        extra_dataitems[k] = mlrun.datastore.store_manager.object(v, key=k)

    return artifact_spec, extra_dataitems


def generate_target_path(item: Artifact, artifact_path, producer):
    # path convention: artifact_path[/{run_name}]/{iter}/{key}.{suffix}
    # todo: add run_id here (vs in the .run() methods), support items dedup (by hash)
    artifact_path = artifact_path or ""
    if artifact_path and not artifact_path.endswith("/"):
        artifact_path += "/"
    if producer.kind == "run":
        artifact_path += f"{producer.name}/{item.iter or 0}/"

    suffix = "/"
    if not item.is_dir:
        # suffixes yields a list of suffixes, e.g. ['.tar', '.gz']
        # join them together to get the full suffix, e.g. '.tar.gz'
        suffix = "".join(pathlib.Path(item.src_path or "").suffixes)
        if not suffix and item.format:
            suffix = f".{item.format}"

    return f"{artifact_path}{item.key}{suffix}"


# TODO: Remove once data migration v5 is obsolete
def convert_legacy_artifact_to_new_format(
    legacy_artifact: dict,
) -> Artifact:
    """Converts a legacy artifact to a new format.
    :param legacy_artifact: The legacy artifact to convert.
    :return: The converted artifact.
    """
    artifact_key = legacy_artifact.get("key", "")
    artifact_tag = legacy_artifact.get("tag", "")
    if artifact_tag:
        artifact_key = f"{artifact_key}:{artifact_tag}"
    # TODO: Remove once data migration v5 is obsolete
    warnings.warn(
        f"Converting legacy artifact '{artifact_key}' to new format. This will not be supported in MLRun 1.10.0. "
        f"Make sure to save the artifact/project in the new format.",
        FutureWarning,
    )

    artifact = mlrun.artifacts.artifact_types.get(
        legacy_artifact.get("kind", "artifact"), mlrun.artifacts.Artifact
    )()

    artifact.metadata = artifact.metadata.from_dict(legacy_artifact)
    artifact.spec = artifact.spec.from_dict(legacy_artifact)
    artifact.status = artifact.status.from_dict(legacy_artifact)

    return artifact


def fill_artifact_object_hash(object_dict, iteration=None, producer_id=None):
    # remove artifact related fields before calculating hash
    object_dict.setdefault("metadata", {})
    labels = object_dict["metadata"].pop("labels", None)
    object_updated_timestamp = object_dict["metadata"].pop("updated", None)

    artifact_cls = mlrun.artifacts.artifact_types.get(
        object_dict.get("kind", "artifact"), Artifact
    )()
    spec_fields_to_exclude = artifact_cls.spec._exclude_fields_from_uid_hash
    spec_fields_to_exclude_values = []
    object_dict.setdefault("spec", {})
    for field in spec_fields_to_exclude:
        spec_fields_to_exclude_values.append(object_dict["spec"].pop(field, None))

    # make sure we have a key, producer_id and iteration, as they determine the artifact uniqueness
    if not object_dict["metadata"].get("key"):
        raise ValueError("Artifact key is not set")
    object_dict["metadata"]["iter"] = iteration or object_dict["metadata"].get("iter")
    object_dict["metadata"]["tree"] = object_dict["metadata"].get("tree") or producer_id

    # calc hash and fill
    uid = mlrun.utils.helpers.fill_object_hash(object_dict, "uid")

    # restore original values
    if labels:
        object_dict["metadata"]["labels"] = labels
    if object_updated_timestamp:
        object_dict["metadata"]["updated"] = object_updated_timestamp
    for key, value in zip(spec_fields_to_exclude, spec_fields_to_exclude_values):
        if value is not None:
            object_dict["spec"][key] = value

    return uid
