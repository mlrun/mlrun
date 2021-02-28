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
import typing

import pandas as pd


import mlrun.errors
import mlrun.api.schemas
from ..features import Feature
from ..model import VersionedObjMetadata
from ..feature_store.common import parse_feature_string, get_feature_set_by_uri
from ..model import ModelObj, ObjectList, DataSource, DataTarget
from ..config import config as mlconf
from ..runtimes.function_reference import FunctionReference
from ..serving.states import RootFlowState
from ..datastore.targets import get_offline_target, ParquetTarget, CSVTarget
from ..datastore import get_store_uri
from ..utils import StorePrefix


class FeatureVectorSpec(ModelObj):
    """
    Spec for FeatureVector
    """

    def __init__(
        self,
        features: typing.Optional[typing.List[str]] = None,
        description: typing.Optional[str] = None,
        entity_source: typing.Optional[typing.Union[dict, DataSource]] = None,
        entity_fields: typing.Optional[typing.List[Feature]] = None,
        timestamp_field: typing.Optional[str] = None,
        graph: typing.Optional[typing.Union[dict, RootFlowState]] = None,
        label_column: typing.Optional[str] = None,
        function: typing.Optional[typing.Union[dict, FunctionReference]] = None,
        analysis: typing.Optional[dict] = None,
    ):
        """
        :param features: Optional, list of features
        :param description: Optional, description
        :param entity_source: Optional, name of data source for features
        :param entity_fields: Optional, the features in the vector
        :param timestamp_field: Optional, timestamp_field
        :param graph: Optional, provide graph (root state)
        :param label_column: Optional, Which column in the is the label (target) column
        :param function: Optional, template graph processing function reference
        :param analysis: Optional, linked artifacts/files for analysis
        """
        self._graph: RootFlowState = None
        self._entity_fields: ObjectList = None
        self._entity_source: DataSource = None
        self._function: FunctionReference = None

        self.description = description
        self.features: typing.List[str] = features or []
        self.entity_source = entity_source
        self.entity_fields = entity_fields or []
        self.graph = graph
        self.timestamp_field = timestamp_field
        self.label_column = label_column
        self.function = function
        self.analysis = analysis or {}

    @property
    def entity_source(self) -> DataSource:
        """data source used as entity source (events/keys need to be enriched)"""
        return self._entity_source

    @entity_source.setter
    def entity_source(self, source: typing.Union[dict, DataSource]):
        """
        Set entity data source
        :param source: datasource as dic or DataSource object
        :return:
        """
        self._entity_source = self._verify_dict(source, "entity_source", DataSource)

    @property
    def entity_fields(self) -> ObjectList:
        """the schema/metadata for the entity source fields"""
        return self._entity_fields

    @entity_fields.setter
    def entity_fields(self, entity_fields: typing.List[Feature]):
        self._entity_fields = ObjectList.from_list(Feature, entity_fields)

    @property
    def graph(self) -> RootFlowState:
        """feature vector transformation graph/DAG"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowState)
        self._graph.engine = "async"

    @property
    def function(self) -> FunctionReference:
        """reference to template graph processing function"""
        return self._function

    @function.setter
    def function(self, function):
        self._function = self._verify_dict(function, "function", FunctionReference)


class FeatureVectorStatus(ModelObj):
    def __init__(
        self,
        state: typing.Optional[str] = None,
        targets: typing.Optional[typing.List[DataTarget]] = None,
        features: typing.Optional[typing.List[Feature]] = None,
        stats: typing.Optional[dict] = None,
        preview: typing.Optional[list] = None,
        run_uri: typing.Optional[str] = None,
    ):
        self._targets: ObjectList = None
        self._features: ObjectList = None

        self.state = state or "created"
        self.targets = targets
        self.stats = stats or {}
        self.preview = preview or []
        self.features: typing.List[Feature] = features or []
        self.run_uri = run_uri

    @property
    def targets(self) -> ObjectList:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: typing.List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    @property
    def features(self) -> ObjectList:
        """list of features (result of joining features from the source feature sets)"""
        return self._features

    @features.setter
    def features(self, features: typing.List[Feature]):
        self._features = ObjectList.from_list(Feature, features)


class FeatureVector(ModelObj):
    """
    Feature vector, specifies selected features, their metadata and material views
    """

    kind = mlrun.api.schemas.ObjectKind.feature_vector.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name: typing.Optional[str] = None,
        features: typing.Optional[typing.List[str]] = None,
        description: typing.Optional[str] = None,
    ) -> None:
        """

        :param name: Optional, name for the vector
        :param features: Optional, list of features (strings)
        :param description: Optional, description of the vector
        """
        self._spec: FeatureVectorSpec = None
        self._metadata = None
        self._status = None

        self.spec = FeatureVectorSpec(description=description, features=features)
        self.metadata = VersionedObjMetadata(name=name)
        self.status = None

        self._entity_df = None
        self._feature_set_fields = {}
        self.feature_set_objects = {}

    @property
    def spec(self) -> FeatureVectorSpec:
        """
        Get the feature vector spec

        :return: Feature vector spec
        """
        return self._spec

    @spec.setter
    def spec(self, spec: typing.Union[dict, FeatureVectorSpec]):
        """
        Set the feature vector spec

        :param spec: spec dict or object
        """
        self._spec = self._verify_dict(spec, "spec", FeatureVectorSpec)

    @property
    def metadata(self) -> VersionedObjMetadata:
        """
        Get feature vector metadata

        :return: metadata object
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: typing.Union[dict, VersionedObjMetadata]):
        """
        Set feature vector metadata from dict or object

        :param metadata: dict or object representing metadata
        """
        self._metadata = self._verify_dict(metadata, "metadata", VersionedObjMetadata)

    @property
    def status(self) -> FeatureVectorStatus:
        """
        Get the feature vector status

        :return: Feature vector status object
        """
        return self._status

    @status.setter
    def status(self, status: typing.Union[dict, FeatureVectorStatus]):
        """
        Set feature vector status from dict or status object

        :param status: status as dict or object
        """
        self._status = self._verify_dict(status, "status", FeatureVectorStatus)

    @property
    def uri(self) -> str:
        """
        Get the fully qualified feature vector uri

        :return: uri
        """
        uri = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        uri = get_store_uri(StorePrefix.FeatureVector, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def link_analysis(self, name, uri):
        """
        Add a linked file/artifact (chart, data, ..)

        :param name: artifact name
        :param uri: artifact full uri
        """
        self._spec.analysis[name] = uri

    def get_stats_table(self) -> pd.DataFrame:
        """
        Get feature vector statistics table (as dataframe)

        :return: New statistics DataFrame object
        """
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def get_target_path(self, name: typing.Optional[str] = None):
        """
        Get the url/path for an offline or specified data target

        :param name: Optional, name of specific data target
        :return Target path
        """
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def to_dataframe(
        self, df_module=None, target_name=None
    ) -> typing.Union[pd.DataFrame, typing.Any]:
        """
        Return feature vector (offline) data as dataframe

        :param df_module: Optional, dataframe class (e.g. pd, dd, cudf, ..)
        :param target_name: Optional, name of the target to take feature vector from

        :return: Dataframe object (possibly of the df_module, of pandas by default)
        """
        driver = get_offline_target(self, name=target_name)
        if not driver:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature vector"
            )
        return driver.as_df(df_module=df_module)

    def save(self, tag: str = "", versioned: bool = False):
        """
        Save the feature vector to mlrun db

        :param tag: The ``tag`` of the object to set in the DB, for example ``latest``.
        :param versioned: Whether to maintain versions for this feature vector. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        """
        db = mlrun.get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        db.store_feature_vector(as_dict, tag=tag, versioned=versioned)

    def reload(self, update_spec=True):
        """
        Reload/sync the feature vector status and spec from the MLRun DB

        :param update_spec: Whether to update the spec (and not only the status) from DB
        """
        from_db = mlrun.get_run_db().get_feature_vector(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        self.status = from_db.status
        if update_spec:
            self.spec = from_db.spec

    def parse_features(self):
        """
        Parse and validate feature list (from vector) and add metadata from feature sets

        :returns
            feature_set_objects: cache of used feature set objects
            feature_set_fields:  list of field (name, alias) per featureset
        """
        processed_features = {}  # dict of name to (featureset, feature object)
        feature_set_objects = {}
        feature_set_fields = collections.defaultdict(list)

        def add_feature(_name, _alias, _feature_set_object):
            if alias in processed_features.keys():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"feature name/alias {alias} already specified,"
                    " use another alias (feature-set:name[@alias])"
                )
            _feature = _feature_set_object[_name]
            processed_features[_alias or name] = (_feature_set_object, _feature)
            _feature_set_name = _feature_set_object.metadata.name
            feature_set_fields[_feature_set_name].append((_name, _alias))

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
    """
    get_online_feature_service response object
    """

    def __init__(self, vector, graph) -> None:
        self.vector = vector
        self._controller = graph.controller

    @property
    def status(self) -> str:
        """
        Get vector prep function status.

        :return: Status - One of [ready, running, error]
        """
        return "ready"

    def get(self, entity_rows: typing.List[dict]) -> typing.List[typing.Any]:
        """
        Get feature vector given the provided entity inputs
        """
        results = []
        futures = []
        for row in entity_rows:
            futures.append(self._controller.emit(row, return_awaitable_result=True))
        for future in futures:
            result = future.await_result()
            results.append(result.body)

        return results

    def close(self):
        """
        Terminate the async loop
        """
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
        if self.status != "completed":
            raise mlrun.errors.MLRunTaskNotReady("feature vector dataset is not ready")
        return self._merger.get_df()

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        return ParquetTarget(path=target_path).write_dataframe(
            self._merger.get_df(), **kw
        )

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        return CSVTarget(path=target_path).write_dataframe(self._merger.get_df(), **kw)
