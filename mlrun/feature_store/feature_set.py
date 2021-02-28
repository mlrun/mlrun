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
import typing

import mlrun.errors
import mlrun.api.schemas
import pandas as pd

from ..db import get_run_db
from ..features import Feature, Entity
from ..model import VersionedObjMetadata
from ..datastore.targets import get_offline_target, default_target_names, TargetTypes, BaseStoreTarget
from ..model import ModelObj, ObjectList, DataSource, DataTarget, DataTargetBase
from ..runtimes.function_reference import FunctionReference
from ..serving.states import BaseState, RootFlowState, previous_step
from ..config import config as mlconf
from ..utils import StorePrefix
from ..datastore import get_store_uri

aggregates_step = "Aggregates"


class FeatureAggregation(ModelObj):
    """
    Feature aggregation requirements
    """

    def __init__(
        self,
            name: typing.Optional[str] = None,
            column: typing.Optional[str] = None,
            operations: typing.Optional[typing.List[str]] = None,
            windows: typing.Optional[typing.List[str]] = None,
            period: typing.Optional[str] = None,
    ) -> None:
        """
        :param name:       Aggregation name/prefix
        :param column:     Name of column/field aggregate
        :param operations: Aggregation operations, e.g. ['sum', 'std']
        :param windows:    List of time windows, e.g. ['1h', '6h', '1d']
        :param period:     Optional, sliding window granularity, e.g. '10m'
        """
        self.name = name
        self.column = column
        self.operations = operations or []
        self.windows = windows or []
        self.period = period


class FeatureSetSpec(ModelObj):
    """
    Spec for FeatureSet
    """
    def __init__(
        self,
        owner: typing.Optional[str] = None,
        description: typing.Optional[str] = None,
        entities: typing.Optional[typing.List[Entity]] = None,
        features: typing.Optional[typing.List[Feature]] = None,
        partition_keys: typing.Optional[typing.List[str]] = None,
        timestamp_key: typing.Optional[str] = None,
        label_column: typing.Optional[str] = None,
        relations: typing.Optional[dict] = None,
        source: typing.Optional[typing.Union[dict, DataSource]] = None,
        targets: typing.Optional[typing.List[DataTargetBase]] = None,
        graph: typing.Optional[typing.Union[dict, RootFlowState]] = None,
        function: typing.Optional[typing.Union[dict, FunctionReference]] = None,
        analysis: typing.Optional[dict] = None,
        engine: typing.Optional[str] = None,
    ):
        """

        :param owner: Optional, owner of the feature set
        :param description: Optional, description
        :param entities: Optional, list data entities
        :param features: Optional, list of features
        :param partition_keys: Optional, list of partition keys
        :param timestamp_key: Optional, timestamp_key
        :param label_column: Optional, Which column in the is the label (target) column
        :param relations: Optional, relationships
        :param source: Optional, specify data source
        :param targets: Optional, specify data targets
        :param graph: Optional, provide graph (root state)
        :param function: Optional, template graph processing function reference
        :param analysis: Optional, linked artifacts/files for analysis
        :param engine: Optional, Specify graph engine (defaults to "async")
        """
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._graph: RootFlowState = None
        self._source = None
        self._function: FunctionReference = None

        self.owner = owner
        self.description = description
        self.entities: typing.List[Entity] = entities or []
        self.features: typing.List[Feature] = features or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.relations = relations or {}
        self.source = source
        self.targets = targets or []
        self.graph = graph
        self.label_column = label_column
        self.function = function
        self.analysis = analysis or {}
        self.engine = engine

    @property
    def entities(self) -> ObjectList:
        """
        Feature set entities (indexes)

        :return: object list of entities
        """
        return self._entities

    @entities.setter
    def entities(self, entities: typing.List[Entity]):
        """
        Set feature set entities (indexes)

        :param entities: entities to set
        """
        self._entities = ObjectList.from_list(Entity, entities)

    @property
    def features(self) -> ObjectList:
        """
        Feature set features list

        :return: object list of features
        """
        return self._features

    @features.setter
    def features(self, features: typing.List[Feature]):
        """
        Set the feature set list
        :param features: list of Feature objects
        """
        self._features = ObjectList.from_list(Feature, features)

    @property
    def targets(self) -> ObjectList:
        """
        Get list object of desired targets (material storage)

        :return object list of targets
        """
        return self._targets

    @targets.setter
    def targets(self, targets: typing.List[DataTargetBase]):
        """
        Set targets list

        :param targets: list of DataTarget objects
        """
        self._targets = ObjectList.from_list(DataTargetBase, targets)

    @property
    def graph(self) -> RootFlowState:
        """
        Get feature set transformation graph/DAG
        """
        return self._graph

    @graph.setter
    def graph(self, graph: typing.Union[dict, RootFlowState]):
        """
        Set the transformation graph for the feature set

        :param graph: graph dict or root state object
        """
        self._graph = self._verify_dict(graph, "graph", RootFlowState)
        self._graph.engine = "async"

    @property
    def function(self) -> FunctionReference:
        """
        Get template graph processing function reference

        :return: function reference
        """
        return self._function

    @function.setter
    def function(self, function: typing.Union[dict, FunctionReference]):
        """
        Set function reference for template graph processing

        :param function: function reference object
        """
        self._function = self._verify_dict(function, "function", FunctionReference)

    @property
    def source(self) -> DataSource:
        """
        Get feature set data source definitions

        :return: data source
        """
        return self._source

    @source.setter
    def source(self, source: typing.Union[dict, DataSource]):
        """
        Set source definition for feature set

        :param source: source definition
        """
        self._source = self._verify_dict(source, "source", DataSource)

    def require_processing(self) -> bool:
        """
        Is processing required on this feature set
        :return: bool (true if there are any graph states)
        """
        return len(self._graph.states) > 0


class FeatureSetStatus(ModelObj):
    """
    Status for FeatureSet
    """
    def __init__(
        self,
        state: typing.Optional[str] = None,
        targets: typing.Optional[typing.List[DataTarget]] = None,
        stats: typing.Optional[dict] = None,
        preview: typing.Optional[list] = None,
        function_uri: typing.Optional[str] = None,
        run_uri: typing.Optional[str] = None,
    ):
        self.state = state or "created"
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = preview or []
        self.function_uri = function_uri
        self.run_uri = run_uri

    @property
    def targets(self) -> ObjectList:
        """
        list of material storage targets + their status/path

        :return: Object list of DataTarget
        """
        return self._targets

    @targets.setter
    def targets(self, targets: typing.List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        """
        Update targets with given target

        :param target: data target object to update
        """
        self._targets.update(target)


class FeatureSet(ModelObj):
    """
    Feature set object, defines a set of features and their data pipeline
    """
    kind = mlrun.api.schemas.ObjectKind.feature_set.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self,
                 name: str = None,
                 description: str = None,
                 entities: typing.Optional[typing.List[Entity]] = None,
                 timestamp_key: typing.Optional[str] = None):
        """
        :param name: Optional, name
        :param description: Optional, description
        :param entities: Optional, list of data entities
        :param timestamp_key: Optional, key of the timestamp feature
        """
        self._spec: FeatureSetSpec = None
        self._metadata: VersionedObjMetadata = None
        self._status: FeatureSetStatus = None

        self.spec = FeatureSetSpec(
            description=description, entities=entities, timestamp_key=timestamp_key
        )
        self.metadata = VersionedObjMetadata(name=name)
        self.status = None
        self._last_state = ""

    @property
    def spec(self) -> FeatureSetSpec:
        """
        Get the feature set spec

        :return: Feature set spec
        """
        return self._spec

    @spec.setter
    def spec(self, spec: typing.Union[dict, FeatureSetSpec]):
        """
        Set feature set spec

        :param spec: Feature set spec to set
        """
        self._spec = self._verify_dict(spec, "spec", FeatureSetSpec)

    @property
    def metadata(self) -> VersionedObjMetadata:
        """
        Get feature set metadata

        :return: Feature set metadata
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: typing.Union[dict, VersionedObjMetadata]):
        """
        Set feature set metadata

        :param metadata: Feature set metadata
        """
        self._metadata = self._verify_dict(metadata, "metadata", VersionedObjMetadata)

    @property
    def status(self) -> FeatureSetStatus:
        """
        Get feature set status

        :return: feature set status
        """
        return self._status

    @status.setter
    def status(self, status: typing.Union[dict, FeatureSetStatus]):
        """
        Set feature set status
        :param status: Status to set
        """
        self._status = self._verify_dict(status, "status", FeatureSetStatus)

    @property
    def uri(self) -> str:
        """
        Get the fully qualified feature set uri

        :return: uri
        """
        uri = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        uri = get_store_uri(StorePrefix.FeatureSet, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def get_target_path(self, name: typing.Optional[str] = None) -> str:
        """
        Get the url/path for an offline or specified data target

        :param name: Optional, name of specific data target
        :return Target path
        """
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def set_targets(self,
                    targets: typing.Optional[typing.List[typing.Union[str, BaseStoreTarget]]] = None,
                    with_defaults: bool = True):
        """
        Set the desired target list or defaults

        :param targets:  List of target type names ('csv', 'nosql', ..) or target objects
                         CSVTarget(), ParquetTarget(), NoSqlTarget(), ..
        :param with_defaults: Whether to add the default targets (as defined in the central config)
        """
        if targets is not None and not isinstance(targets, list):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "targets can only be None or a list of kinds or DataTargetBase derivatives"
            )
        targets = targets or []
        if with_defaults:
            targets.extend(default_target_names())
        for target in targets:
            kind = target.kind if hasattr(target, "kind") else target
            if kind not in TargetTypes.all():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"target kind is not supported, use one of: {','.join(TargetTypes.all())}"
                )
            if not hasattr(target, "kind"):
                target = DataTargetBase(target, name=str(target))
            self.spec.targets.update(target)

    def add_entity(self, entity: Entity, name: typing.Optional[str] = None):
        """
        Add/set an entity

        :param entity: Entity to add
        :param name: Optional name to set on the entity
        """
        self._spec.entities.update(entity, name)

    def add_feature(self, feature: Feature, name: typing.Optional[str] = None):
        """
        Add/set a feature

        :param feature: Feature to add
        :param name: Optional name to set on the feature
        """
        self._spec.features.update(feature, name)

    def link_analysis(self, name: str, uri: str):
        """
        Add a linked file/artifact (chart, data, ..)

        :param name: artifact name
        :param uri: artifact full uri
        """
        self._spec.analysis[name] = uri

    @property
    def graph(self) -> RootFlowState:
        """
        Get feature set transformation graph/DAG
        """
        return self.spec.graph

    def add_aggregation(
        self,
        name: str,
        column: str,
        operations: typing.List[str],
        windows: typing.List[str],
        period: typing.Optional[str] = None,
        state_name: typing.Optional[str] = None,
        after: typing.Optional[str] = None,
        before: typing.Optional[str] = None,
    ):
        """
        Add feature aggregation rule

        example::

            myset.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")

        :param name:       Aggregation name/prefix
        :param column:     Name of column/field aggregate
        :param operations: Aggregation operations, e.g. ['sum', 'std']
        :param windows:    List of time windows, e.g. ['1h', '6h', '1d']
        :param period:     Optional, sliding window granularity, e.g. '10m'
        :param state_name: Optional, graph state name
        :param after:      Optional, after which graph state it runs
        :param before:     Optional, comes before graph state
        """
        aggregation = FeatureAggregation(
            name, column, operations, windows, period
        ).to_dict()

        def upsert_feature(name):
            if name in self.spec.features:
                self.spec.features[name].aggregate = True
            else:
                self.spec.features[name] = Feature(name=column, aggregate=True)

        state_name = state_name or aggregates_step
        graph = self.spec.graph
        if state_name in graph.states:
            state = graph.states[state_name]
            aggregations = state.class_args.get("aggregates", [])
            aggregations.append(aggregation)
            state.class_args["aggregates"] = aggregations
        else:
            graph.add_step(
                name=state_name,
                after=after or previous_step,
                before=before,
                class_name="storey.AggregateByKey",
                aggregates=[aggregation],
                table=".",
            )

        for operation in operations:
            for window in windows:
                upsert_feature(f"{name}_{operation}_{window}")

    def get_stats_table(self) -> pd.DataFrame:
        """
        Get feature statistics table (as dataframe)

        :return: New statistics DataFrame object
        """
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def __getitem__(self, name):
        return self._spec.features[name]

    def __setitem__(self, key, item):
        self._spec.features.update(item, key)

    def plot(self,
             filename: typing.Optional[str] = None,
             format: typing.Optional[str] = None,
             with_targets: bool = False,
             **kw):
        """
        Generate graphviz plot

        :param filename: Filename for saving the source of the graphviz graph
        :param format: file extension/format
        :param with_targets: Whether to plot all targets as well

        :return: The generated graph object (graphviz must be installed)
        """
        graph = self.spec.graph
        _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)
        targets = None
        if with_targets:
            targets = [
                BaseState(
                    target.kind,
                    after=target.after_state or default_final_state,
                    shape="cylinder",
                )
                for target in self.spec.targets
            ]
        return graph.plot(filename, format, targets=targets, **kw)

    def to_dataframe(self,
                     columns: typing.Optional[typing.List[str]] = None,
                     df_module:  typing.Optional[typing.Any] = None,
                     target_name: typing.Optional[str] = None) -> typing.Union[pd.DataFrame, typing.Any]:
        """
        Return featureset (offline) data as dataframe

        :param columns:   Optional, list of columns to select
        :param df_module: Optional, dataframe class (e.g. pd, dd, cudf, ..)
        :param target_name: Optional, name of the target to take featureset from

        :return: Dataframe object (possibly of the df_module, of pandas by default)
        """
        if columns:
            entities = list(self.spec.entities.keys())
            if self.spec.timestamp_key and self.spec.timestamp_key not in entities:
                columns = [self.spec.timestamp_key] + columns
            columns = entities + columns
        driver = get_offline_target(self, name=target_name)
        if not driver:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature set"
            )
        return driver.as_df(columns=columns, df_module=df_module)

    def save(self, tag: str = "", versioned: bool = False):
        """
        Save the feature set to mlrun db

        :param tag: The ``tag`` of the object to set in the DB, for example ``latest``.
        :param versioned: Whether to maintain versions for this feature set. All versions of a versioned object
            will be kept in the DB and can be retrieved until explicitly deleted.
        """
        db = get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        as_dict["spec"]["features"] = as_dict["spec"].get(
            "features", []
        )  # bypass DB bug
        db.store_feature_set(as_dict, tag=tag, versioned=versioned)

    def reload(self, update_spec: bool = True):
        """
        Reload/sync the feature set status and spec from the MLRun DB

        :param update_spec: Whether to update the spec (and not only the status) from DB
        """
        feature_set = get_run_db().get_feature_set(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        self.status = feature_set.status
        if update_spec:
            self.spec = feature_set.spec
