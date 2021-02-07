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
import collections
from typing import List
import mlrun
import pandas as pd


from ...features import Feature
from .base import DataSource, DataTarget, CommonMetadata
from ..common import parse_feature_string, get_feature_set_by_uri
from ...model import ModelObj, ObjectList
from ...artifacts.dataset import upload_dataframe
from ...config import config as mlconf
from ...serving.states import RootFlowState
from ..targets import get_offline_target
from ...datastore import get_store_uri
from ...utils import StorePrefix


class FeatureVectorSpec(ModelObj):
    def __init__(
        self,
        features=None,
        description=None,
        entity_source=None,
        entity_fields=None,
        timestamp_field=None,
        graph=None,
        label_column=None,
        analysis=None,
    ):
        self._graph: RootFlowState = None
        self._entity_fields: ObjectList = None
        self._entity_source: DataSource = None

        self.description = description
        self.features: List[str] = features or []
        self.entity_source = entity_source
        self.entity_fields = entity_fields or []
        self.graph = graph
        self.timestamp_field = timestamp_field
        self.label_column = label_column
        self.analysis = analysis or {}

    @property
    def entity_source(self) -> DataSource:
        """data source used as entity source (events/keys need to be enriched)"""
        return self._entity_source

    @entity_source.setter
    def entity_source(self, source: DataSource):
        self._entity_source = self._verify_dict(source, "entity_source", DataSource)

    @property
    def entity_fields(self) -> List[Feature]:
        """the schema/metadata for the entity source fields"""
        return self._entity_fields

    @entity_fields.setter
    def entity_fields(self, entity_fields: List[Feature]):
        self._entity_fields = ObjectList.from_list(Feature, entity_fields)

    @property
    def graph(self) -> RootFlowState:
        """feature vector transformation graph/DAG"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowState)
        self._graph.engine = "async"


class FeatureVectorStatus(ModelObj):
    def __init__(
        self, state=None, targets=None, features=None, stats=None, preview=None
    ):
        self._targets: ObjectList = None
        self._features: ObjectList = None

        self.state = state or "created"
        self.targets = targets
        self.stats = stats or {}
        self.preview = preview or []
        self.features: List[Feature] = features or []

    @property
    def targets(self) -> List[DataTarget]:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    @property
    def features(self) -> List[Feature]:
        """list of features (result of joining features from the source feature sets)"""
        return self._features

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)


class FeatureVector(ModelObj):
    """Feature vector, specify selected features, their metadata and material views"""

    kind = kind = mlrun.api.schemas.ObjectKind.feature_vector.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, features=None, description=None):
        self._spec: FeatureVectorSpec = None
        self._metadata = None
        self._status = None

        self.spec = FeatureVectorSpec(description=description, features=features)
        self.metadata = CommonMetadata(name=name)
        self.status = None

        self._entity_df = None
        self._feature_set_fields = {}
        self.feature_set_objects = {}

    @property
    def spec(self) -> FeatureVectorSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureVectorSpec)

    @property
    def metadata(self) -> CommonMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", CommonMetadata)

    @property
    def status(self) -> FeatureVectorStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureVectorStatus)

    @property
    def uri(self):
        """fully qualified feature vector uri"""
        uri = f'{self._metadata.project or ""}/{self._metadata.name}'
        uri = get_store_uri(StorePrefix.FeatureVector, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def link_analysis(self, name, uri):
        """add a linked file/artifact (chart, data, ..)"""
        self._spec.analysis[name] = uri

    def get_stats_table(self):
        """get feature statistics table (as dataframe)"""
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def get_target_path(self, name=None):
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def to_dataframe(self, df_module=None, target_name=None):
        """return feature vector (offline) data as dataframe"""
        driver = get_offline_target(self, name=target_name)
        if not driver:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature vector"
            )
        return driver.as_df(df_module=df_module)

    def save(self, tag="", versioned=False):
        """save to mlrun db"""
        db = mlrun.get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        db.store_feature_vector(as_dict, tag=tag, versioned=versioned)

    def parse_features(self):
        """parse and validate feature list (from vector) and add metadata from feature sets

        :returns
            feature_set_objects: cache of used feature set objects
            feature_set_fields:  list of field (name, alias) per featureset
        """
        processed_features = {}  # dict of name to (featureset, feature object)
        feature_set_objects = {}
        feature_set_fields = collections.defaultdict(list)

        def add_feature(name, alias, feature_set_object):
            if alias in processed_features.keys():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"feature name/alias {alias} already specified,"
                    " use another alias (feature-set:name[@alias])"
                )
            feature = feature_set_object[name]
            processed_features[alias or name] = (feature_set_object, feature)
            featureset_name = feature_set_object.metadata.name
            feature_set_fields[featureset_name].append((name, alias))

        for feature in self.spec.features:
            feature_set, feature_name, alias = parse_feature_string(feature)
            if feature_set not in feature_set_objects.keys():
                feature_set_objects[feature_set] = get_feature_set_by_uri(
                    feature_set, self.metadata.project
                )
            feature_set_object = feature_set_objects[feature_set]

            feature_fields = feature_set_object.spec.features.keys()
            if feature_name == "*":
                for field in feature_fields:
                    if field != feature_set_object.spec.timestamp_key:
                        if alias:
                            add_feature(field, alias + "_" + field, feature_set_object)
                        else:
                            add_feature(field, field, feature_set_object)
            else:
                if feature_name not in feature_fields:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"feature {feature} not found in feature set {feature_set}"
                    )
                add_feature(feature_name, alias, feature_set_object)

        for feature_set_name, fields in feature_set_fields.items():
            feature_set = feature_set_objects[feature_set_name]
            for name, alias in fields:
                field_name = alias or name
                if name in feature_set.status.stats:
                    self.status.stats[field_name] = feature_set.status.stats[name]
                if name in feature_set.spec.features.keys():
                    self.status.features[field_name] = feature_set.spec.features[name]

        return feature_set_objects, feature_set_fields


class OnlineVectorService:
    """get_online_feature_service response object"""

    def __init__(self, vector, graph):
        self.vector = vector
        self._controller = graph.controller

    @property
    def status(self):
        """vector prep function status (ready, running, error)"""
        return "ready"

    def get(self, entity_rows: List[dict]):
        """get feature vector given the provided entity inputs"""
        results = []
        futures = []
        for row in entity_rows:
            futures.append(self._controller.emit(row, return_awaitable_result=True))
        for future in futures:
            result = future.await_result()
            results.append(result.body)

        return results

    def close(self):
        """terminate the async loop"""
        self._controller.terminate()


class OfflineVectorResponse:
    """get_offline_features response object"""

    def __init__(self, merger):
        self._merger = merger
        self.vector = merger.vector

    @property
    def status(self):
        """vector prep job status (ready, running, error)"""
        return self._merger.get_status()

    def to_dataframe(self):
        """return result as dataframe"""
        if self.status != "ready":
            raise mlrun.errors.MLRunTaskNotReady("feature vector dataset is not ready")
        return self._merger.get_df()

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        return self._upload(target_path, "parquet", **kw)

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        return self._upload(target_path, "csv", **kw)

    def _upload(self, target_path, format="parquet", src_path=None, **kw):
        upload_dataframe(
            self._merger.get_df(), target_path, format=format, src_path=src_path, **kw,
        )
