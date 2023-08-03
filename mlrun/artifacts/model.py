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
import tempfile
from os import path
from typing import List

import yaml
from deprecated import deprecated

import mlrun

from ..data_types import InferOptions, get_infer_interface
from ..datastore import is_store_uri, store_manager
from ..features import Feature
from ..model import ObjectList
from ..utils import StorePrefix, is_relative_path
from .base import Artifact, ArtifactSpec, LegacyArtifact, upload_extra_data

model_spec_filename = "model_spec.yaml"


class ModelArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "model_file",
        "metrics",
        "parameters",
        "inputs",
        "outputs",
        "framework",
        "algorithm",
        "feature_vector",
        "feature_weights",
        "feature_stats",
        "model_target_file",
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
        model_file=None,
        metrics=None,
        paraemeters=None,
        inputs: List[Feature] = None,
        outputs: List[Feature] = None,
        framework=None,
        algorithm=None,
        feature_vector=None,
        feature_weights=None,
        feature_stats=None,
        model_target_file=None,
    ):
        super().__init__(
            src_path,
            target_path,
            viewer,
            is_inline,
            format,
            size,
            db_key,
            extra_data,
            body,
        )
        self.model_file = model_file
        self.metrics = metrics or {}
        self.parameters = paraemeters or {}
        self.inputs: List[Feature] = inputs or []
        self.outputs: List[Feature] = outputs or []
        self.framework = framework
        self.algorithm = algorithm
        self.feature_vector = feature_vector
        self.feature_weights = feature_weights
        self.feature_stats = feature_stats
        self.model_target_file = model_target_file

    @property
    def inputs(self) -> List[Feature]:
        """input feature list"""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[Feature]):
        self._inputs = ObjectList.from_list(Feature, inputs)

    @property
    def outputs(self) -> List[Feature]:
        """output feature list"""
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[Feature]):
        self._outputs = ObjectList.from_list(Feature, outputs)


class ModelArtifact(Artifact):
    """ML Model artifact

    Store link to ML model file(s) along with the model metrics, parameters, schema, and stats
    """

    kind = "model"
    _store_prefix = StorePrefix.Model

    def __init__(
        self,
        key=None,
        body=None,
        format=None,
        model_file=None,
        metrics=None,
        target_path=None,
        parameters=None,
        inputs=None,
        outputs=None,
        framework=None,
        algorithm=None,
        feature_vector=None,
        feature_weights=None,
        extra_data=None,
        model_dir=None,
        **kwargs,
    ):

        super().__init__(key, body, format=format, target_path=target_path, **kwargs)
        if model_file and "://" in model_file:
            model_dir = path.dirname(model_file)
            model_file = path.basename(model_file)

        self.spec.model_file = model_file
        self.spec.src_path = model_dir
        self.spec.parameters = parameters or {}
        self.spec.metrics = metrics or {}
        self.spec.inputs = inputs or []
        self.spec.outputs = outputs or []
        self.spec.extra_data = extra_data or {}
        self.spec.framework = framework
        self.spec.algorithm = algorithm
        self.spec.feature_vector = feature_vector
        self.spec.feature_weights = feature_weights
        self.spec.feature_stats = None

    @property
    def spec(self) -> ModelArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ModelArtifactSpec)

    @property
    def inputs(self) -> List[Feature]:
        """input feature list"""
        return self.spec.inputs

    @inputs.setter
    def inputs(self, inputs: List[Feature]):
        """input feature list"""
        self.spec.inputs = inputs

    @property
    def outputs(self) -> List[Feature]:
        """input feature list"""
        return self.spec.outputs

    @outputs.setter
    def outputs(self, outputs: List[Feature]):
        """input feature list"""
        self.spec.outputs = outputs

    @property
    def model_file(self):
        return self.spec.model_file

    @model_file.setter
    def model_file(self, model_file):
        self.spec.model_file = model_file

    @property
    def parameters(self):
        return self.spec.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.spec.parameters = parameters

    @property
    def metrics(self):
        return self.spec.metrics

    @metrics.setter
    def metrics(self, metrics):
        self.spec.metrics = metrics

    @property
    def feature_stats(self):
        return self.spec.feature_stats

    @feature_stats.setter
    def feature_stats(self, feature_stats):
        self.spec.feature_stats = feature_stats

    @property
    def feature_vector(self):
        return self.spec.feature_vector

    @feature_vector.setter
    def feature_vector(self, feature_vector):
        self.spec.feature_vector = feature_vector

    @property
    def feature_weights(self):
        return self.spec.feature_weights

    @feature_weights.setter
    def feature_weights(self, feature_weights):
        self.spec.feature_weights = feature_weights

    @property
    def model_target_file(self):
        return self.spec.model_target_file

    @model_target_file.setter
    def model_target_file(self, model_target_file):
        self.spec.model_target_file = model_target_file

    def infer_from_df(self, df, label_columns=None, with_stats=True, num_bins=None):
        """infer inputs, outputs, and stats from provided df (training set)

        :param df:      dataframe to infer from
        :param label_columns: name of the label (target) column
        :param with_stats:    infer statistics (min, max, .. histogram)
        :param num_bins:      number of bins for histogram
        """
        subset = df
        inferer = get_infer_interface(subset)
        if label_columns:
            if not isinstance(label_columns, list):
                label_columns = [label_columns]
            subset = df.drop(columns=label_columns)
        inferer.infer_schema(
            subset, self.spec.inputs, {}, options=InferOptions.Features
        )
        if label_columns:
            inferer.infer_schema(
                df[label_columns], self.spec.outputs, {}, options=InferOptions.Features
            )
        if with_stats:
            self.spec.feature_stats = inferer.get_stats(
                df, options=InferOptions.Histogram, num_bins=num_bins
            )

    @property
    def is_dir(self):
        return True

    def before_log(self):
        if not self.spec.model_file:
            raise ValueError("model_file attr must be specified")

        super(ModelArtifact, self).before_log()

        if self.spec.framework:
            self.metadata.labels = self.metadata.labels or {}
            self.metadata.labels["framework"] = self.spec.framework

    def upload(self, artifact_path: str = None):
        """
        internal, upload to target store
        :param artifact_path: required only for when generating target_path from artifact hash
        """
        # if mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash outputs True and the user
        # didn't pass target_path explicitly, then target_path will be calculated right before uploading the artifact
        # using `resolve_<body/file>_target_hash_path`
        target_model_path = (
            path.join(self.spec.target_path, self.spec.model_file)
            if self.spec.target_path
            else None
        )
        body = self.spec.get_body()
        if body:
            if not target_model_path:
                (
                    self.metadata.hash,
                    target_model_path,
                ) = self.resolve_body_target_hash_path(
                    body=body, artifact_path=artifact_path
                )
            self._upload_body(
                body, target=target_model_path, artifact_path=artifact_path
            )
        else:
            src_model_path = _get_src_path(self, self.spec.model_file)
            if not path.isfile(src_model_path):
                raise ValueError(f"model file {src_model_path} not found")

            if not target_model_path:
                (
                    self.metadata.hash,
                    target_model_path,
                ) = self.resolve_file_target_hash_path(
                    source_path=src_model_path, artifact_path=artifact_path
                )

            self._upload_file(
                src_model_path,
                target_path=target_model_path,
                artifact_path=artifact_path,
            )

        upload_extra_data(
            artifact=self, extra_data=self.spec.extra_data, artifact_path=artifact_path
        )

        # the model spec yaml should not include the tag, as the same model can be used with different tags,
        # and the tag is not part of the model spec but the metadata of the model artifact
        spec_body = _remove_tag_from_spec_yaml(self)
        spec_target_path = None

        if mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash:

            # resolving target_path for the model spec
            _, spec_target_path = self.resolve_body_target_hash_path(
                body=spec_body, artifact_path=artifact_path
            )

            # if mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash outputs True, then target_path will be
            # will point to the artifact path which is where the model and all its extra data are stored
            self.spec.target_path = (
                artifact_path + "/"
                if not artifact_path.endswith("/")
                else artifact_path
            )
            # unlike in extra_data, which stores for each key the path to the file, in target_path we store the
            # target path dir, and because we generated the target path of the model from the artifact hash,
            # the model_file doesn't represent the actual target file name of the model, so we need to update it
            self.spec.model_target_file = target_model_path.split("/")[-1]

        spec_target_path = spec_target_path or path.join(
            self.spec.target_path, model_spec_filename
        )
        store_manager.object(url=spec_target_path).put(spec_body)

    def _get_file_body(self):
        body = self.spec.get_body()
        if body:
            return body
        src_model_path = _get_src_path(self, self.spec.model_file)
        if src_model_path and path.isfile(src_model_path):
            with open(src_model_path, "rb") as fp:
                return fp.read()
        target_model_path = path.join(self.spec.target_path, self.spec.model_file)
        return mlrun.get_dataitem(target_model_path).get()


# TODO: remove in 1.6.0
@deprecated(
    version="1.3.0",
    reason="'LegacyModelArtifact' will be removed in 1.6.0, use 'ModelArtifact' instead",
    category=FutureWarning,
)
class LegacyModelArtifact(LegacyArtifact):
    """ML Model artifact

    Store link to ML model file(s) along with the model metrics, parameters, schema, and stats
    """

    _dict_fields = LegacyArtifact._dict_fields + [
        "model_file",
        "metrics",
        "parameters",
        "inputs",
        "outputs",
        "framework",
        "algorithm",
        "extra_data",
        "feature_vector",
        "feature_weights",
        "feature_stats",
        "model_target_file",
    ]
    kind = "model"
    _store_prefix = StorePrefix.Model

    def __init__(
        self,
        key=None,
        body=None,
        format=None,
        model_file=None,
        metrics=None,
        target_path=None,
        parameters=None,
        inputs=None,
        outputs=None,
        framework=None,
        algorithm=None,
        feature_vector=None,
        feature_weights=None,
        extra_data=None,
        model_target_file=None,
        **kwargs,
    ):

        super().__init__(key, body, format=format, target_path=target_path, **kwargs)
        self._inputs: ObjectList = None
        self._outputs: ObjectList = None

        self.model_file = model_file
        self.parameters = parameters or {}
        self.metrics = metrics or {}
        self.inputs: List[Feature] = inputs or []
        self.outputs: List[Feature] = outputs or []
        self.extra_data = extra_data or {}
        self.framework = framework
        self.algorithm = algorithm
        self.feature_vector = feature_vector
        self.feature_weights = feature_weights
        self.feature_stats = None
        self.model_target_file = model_target_file

    @property
    def inputs(self) -> List[Feature]:
        """input feature list"""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[Feature]):
        self._inputs = ObjectList.from_list(Feature, inputs)

    @property
    def outputs(self) -> List[Feature]:
        """output feature list"""
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[Feature]):
        self._outputs = ObjectList.from_list(Feature, outputs)

    def infer_from_df(self, df, label_columns=None, with_stats=True, num_bins=None):
        """infer inputs, outputs, and stats from provided df (training set)

        :param df:      dataframe to infer from
        :param label_columns: name of the label (target) column
        :param with_stats:    infer statistics (min, max, .. histogram)
        :param num_bins:      number of bins for histogram
        """
        subset = df
        inferer = get_infer_interface(subset)
        if label_columns:
            if not isinstance(label_columns, list):
                label_columns = [label_columns]
            subset = df.drop(columns=label_columns)
        inferer.infer_schema(subset, self.inputs, {}, options=InferOptions.Features)
        if label_columns:
            inferer.infer_schema(
                df[label_columns], self.outputs, {}, options=InferOptions.Features
            )
        if with_stats:
            self.feature_stats = inferer.get_stats(
                df, options=InferOptions.Histogram, num_bins=num_bins
            )

    @property
    def is_dir(self):
        return True

    def before_log(self):
        if not self.model_file:
            raise ValueError("model_file attr must be specified")

        super(LegacyModelArtifact, self).before_log()

        if self.framework:
            self.labels = self.labels or {}
            self.labels["framework"] = self.framework

    def upload(self):

        target_model_path = path.join(self.target_path, self.model_file)
        body = self.get_body()
        if body:
            self._upload_body(body, target=target_model_path)
        else:
            src_model_path = _get_src_path(self, self.model_file)
            if not path.isfile(src_model_path):
                raise ValueError(f"model file {src_model_path} not found")
            self._upload_file(src_model_path, target=target_model_path)

        upload_extra_data(self, self.extra_data)

        spec_path = path.join(self.target_path, model_spec_filename)
        store_manager.object(url=spec_path).put(self.to_yaml())


def _get_src_path(model_spec: ModelArtifact, filename):
    if model_spec.src_path:
        return path.join(model_spec.src_path, filename)
    return filename


def get_model(model_dir, suffix=""):
    """return model file, model spec object, and list of extra data items

    this function will get the model file, metadata, and extra data
    the returned model file is always local, when using remote urls
    (such as v3io://, s3://, store://, ..) it will be copied locally.

    returned extra data dict (of key, DataItem objects) allow reading additional model files/objects
    e.g. use DataItem.get() or .download(target) .as_df() to read

    example::

        model_file, model_artifact, extra_data = get_model(models_path, suffix='.pkl')
        model = load(open(model_file, "rb"))
        categories = extra_data['categories'].as_df()

    :param model_dir:       model dir or artifact path (store://..) or DataItem
    :param suffix:          model filename suffix (when using a dir)

    :returns: model filename, model artifact object, extra data dict

    """
    model_file = ""
    model_spec = None
    extra_dataitems = {}
    suffix = suffix or ".pkl"
    if hasattr(model_dir, "artifact_url"):
        model_dir = model_dir.artifact_url

    if is_store_uri(model_dir):
        model_spec, target = store_manager.get_store_artifact(model_dir)
        if not model_spec or model_spec.kind != "model":
            raise ValueError(f"store artifact ({model_dir}) is not model kind")
        # in case model_target_file is specified, use it, because that means that the actual model target path
        # in the store is different from the local model_file it was generated from
        model_file = _get_file_path(
            target, model_spec.model_target_file or model_spec.model_file
        )
        extra_dataitems = _get_extra(target, model_spec.extra_data)

    elif model_dir.lower().endswith(".yaml"):
        model_spec = _load_model_spec(model_dir)
        model_file = _get_file_path(model_dir, model_spec.model_file)
        extra_dataitems = _get_extra(model_dir, model_spec.extra_data)

    elif model_dir.endswith(suffix):
        model_file = model_dir
    else:
        dirobj = store_manager.object(url=model_dir)
        model_dir_list = dirobj.listdir()
        if model_spec_filename in model_dir_list:
            model_spec = _load_model_spec(path.join(model_dir, model_spec_filename))
            model_file = _get_file_path(model_dir, model_spec.model_file, isdir=True)
            extra_dataitems = _get_extra(model_dir, model_spec.extra_data, is_dir=True)
        else:
            extra_dataitems = _get_extra(
                model_dir, {v: v for v in model_dir_list}, is_dir=True
            )
            for file in model_dir_list:
                if file.endswith(suffix):
                    model_file = path.join(model_dir, file)
                    break
    if not model_file:
        raise ValueError(f"cant resolve model file for {model_dir} suffix{suffix}")

    obj = store_manager.object(url=model_file)
    if obj.kind == "file":
        return model_file, model_spec, extra_dataitems

    temp_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    obj.download(temp_path)
    return temp_path, model_spec, extra_dataitems


def _load_model_spec(spec_path):
    data = store_manager.object(url=spec_path).get()
    spec = yaml.load(data, Loader=yaml.FullLoader)
    return ModelArtifact.from_dict(spec)


def _get_file_path(base_path: str, name: str, isdir=False):
    if not is_relative_path(name):
        return name
    if not isdir:
        base_path = path.dirname(base_path)
    return path.join(base_path, name).replace("\\", "/")


def _get_extra(target, extra_data, is_dir=False):
    extra_dataitems = {}
    for k, v in extra_data.items():
        extra_dataitems[k] = store_manager.object(
            url=_get_file_path(target, v, isdir=is_dir), key=k
        )
    return extra_dataitems


def _remove_tag_from_spec_yaml(model_spec):
    spec_dict = model_spec.to_dict()
    spec_dict["metadata"].pop("tag", None)
    return yaml.dump(spec_dict)


def update_model(
    model_artifact,
    parameters: dict = None,
    metrics: dict = None,
    extra_data: dict = None,
    inputs: List[Feature] = None,
    outputs: List[Feature] = None,
    feature_vector: str = None,
    feature_weights: list = None,
    key_prefix: str = "",
    labels: dict = None,
    write_spec_copy=True,
    store_object: bool = True,
):
    """Update model object attributes

    this method will edit or add attributes to a model object

    example::

        update_model(model_path, metrics={'speed': 100},
                     extra_data={'my_data': b'some text', 'file': 's3://mybucket/..'})

    :param model_artifact:  model artifact object or path (store://..) or DataItem
    :param parameters:      parameters dict
    :param metrics:         model metrics e.g. accuracy
    :param extra_data:      extra data items key, value dict
                            (value can be: path string | bytes | artifact)
    :param inputs:          list of input features (feature vector schema)
    :param outputs:         list of output features (output vector schema)
    :param feature_vector:  feature store feature vector uri (store://feature-vectors/<project>/<name>[:tag])
    :param feature_weights: list of feature weights, one per input column
    :param key_prefix:      key prefix to add to metrics and extra data items
    :param labels:          metadata labels
    :param write_spec_copy: write a YAML copy of the spec to the target dir
    :param store_object:    Whether to store the model artifact updated.
    """

    if hasattr(model_artifact, "artifact_url"):
        model_artifact = model_artifact.artifact_url

    if isinstance(model_artifact, ModelArtifact):
        model_spec = model_artifact
    elif is_store_uri(model_artifact):
        model_spec, _ = store_manager.get_store_artifact(model_artifact)
    else:
        raise ValueError("model path must be a model store object/URL/DataItem")

    if not model_spec or model_spec.kind != "model":
        raise ValueError(f"store artifact ({model_artifact}) is not model kind")

    if parameters:
        for key, val in parameters.items():
            model_spec.parameters[key] = val
    if metrics:
        for key, val in metrics.items():
            model_spec.metrics[key_prefix + key] = val
    if labels:
        for key, val in labels.items():
            model_spec.labels[key] = val
    if inputs:
        model_spec.inputs = inputs
    if outputs:
        model_spec.outputs = outputs
    if feature_weights:
        model_spec.feature_weights = feature_weights
    if feature_vector:
        model_spec.feature_vector = feature_vector

    if extra_data:
        for key, item in extra_data.items():
            if hasattr(item, "target_path"):
                extra_data[key] = item.target_path

        upload_extra_data(model_spec, extra_data, prefix=key_prefix, update_spec=True)

    if write_spec_copy:
        spec_path = path.join(model_spec.target_path, model_spec_filename)

        # the model spec yaml should not include the tag, as the same model can be used with different tags,
        # and the tag is not part of the model spec but the metadata of the model artifact
        model_spec_yaml = _remove_tag_from_spec_yaml(model_spec)
        store_manager.object(url=spec_path).put(model_spec_yaml)

    model_spec.db_key = model_spec.db_key or model_spec.key
    if store_object:
        mlrun.get_run_db().store_artifact(
            model_spec.db_key,
            model_spec.to_dict(),
            model_spec.tree,
            iter=model_spec.iter,
            project=model_spec.project,
        )
    return model_spec
