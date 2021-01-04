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

from typing import List
import mlrun
import pandas as pd


from .base import DataSource, Feature, DataTarget, CommonMetadata, FeatureStoreError
from ...model import ModelObj, ObjectList
from ...artifacts.dataset import upload_dataframe
from ...config import config as mlconf
from ...serving.states import RootFlowState
from ..targets import get_offline_target
from ...datastore import get_store_uri
from ...utils import StorePrefix


class FeatureVectorError(Exception):
    """ feature vector error. """

    def __init__(self, *args, **kwargs):
        pass


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
    """Feature Vector"""

    kind = "FeatureVector"
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
        target, _ = get_offline_target(self, name=name)
        if target:
            return target.path

    def to_dataframe(self, df_module=None, target_name=None):
        """return feature vector (offline) data as dataframe"""
        target, driver = get_offline_target(self, name=target_name)
        if not target:
            raise FeatureStoreError(
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


class OnlineVectorService:
    """get_online_feature_service response object"""

    def __init__(self, vector, controller):
        self.vector = vector
        self._controller = controller

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
            raise FeatureVectorError("feature vector dataset is not ready")
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
