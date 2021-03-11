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

import pandas as pd

import mlrun

from ..config import config as mlconf
from ..datastore import get_store_uri
from ..datastore.targets import (
    TargetTypes,
    default_target_names,
    get_offline_target,
    validate_target_placement,
)
from ..features import Entity, Feature
from ..model import (
    DataSource,
    DataTarget,
    DataTargetBase,
    ModelObj,
    ObjectList,
    VersionedObjMetadata,
)
from ..runtimes.function_reference import FunctionReference
from ..serving.states import BaseState, RootFlowState, previous_step
from ..utils import StorePrefix

aggregates_step = "Aggregates"


class FeatureAggregation(ModelObj):
    """feature aggregation requirements"""

    def __init__(
        self, name=None, column=None, operations=None, windows=None, period=None
    ):
        self.name = name
        self.column = column
        self.operations = operations or []
        self.windows = windows or []
        self.period = period


class FeatureSetSpec(ModelObj):
    def __init__(
        self,
        owner=None,
        description=None,
        entities=None,
        features=None,
        partition_keys=None,
        timestamp_key=None,
        label_column=None,
        relations=None,
        source=None,
        targets=None,
        graph=None,
        function=None,
        analysis=None,
        engine=None,
    ):
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._graph: RootFlowState = None
        self._source = None
        self._function: FunctionReference = None

        self.owner = owner
        self.description = description
        self.entities: List[Entity] = entities or []
        self.features: List[Feature] = features or []
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
    def entities(self) -> List[Entity]:
        """feature set entities (indexes)"""
        return self._entities

    @entities.setter
    def entities(self, entities: List[Entity]):
        self._entities = ObjectList.from_list(Entity, entities)

    @property
    def features(self) -> List[Feature]:
        """feature set features list"""
        return self._features

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)

    @property
    def targets(self) -> List[DataTargetBase]:
        """list of desired targets (material storage)"""
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTargetBase]):
        self._targets = ObjectList.from_list(DataTargetBase, targets)

    @property
    def graph(self) -> RootFlowState:
        """feature set transformation graph/DAG"""
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

    @property
    def source(self) -> DataSource:
        """feature set data source definitions"""
        return self._source

    @source.setter
    def source(self, source: DataSource):
        self._source = self._verify_dict(source, "source", DataSource)

    def require_processing(self):
        return len(self._graph.states) > 0


class FeatureSetStatus(ModelObj):
    def __init__(
        self,
        state=None,
        targets=None,
        stats=None,
        preview=None,
        function_uri=None,
        run_uri=None,
    ):
        self.state = state or "created"
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = preview or []
        self.function_uri = function_uri
        self.run_uri = run_uri

    @property
    def targets(self) -> List[DataTarget]:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)


class FeatureSet(ModelObj):
    """Feature set object, defines a set of features and their data pipeline"""

    kind = mlrun.api.schemas.ObjectKind.feature_set.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, description=None, entities=None, timestamp_key=None):
        self._spec: FeatureSetSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None

        self.spec = FeatureSetSpec(
            description=description, entities=entities, timestamp_key=timestamp_key
        )
        self.metadata = VersionedObjMetadata(name=name)
        self.status = None
        self._last_state = ""

    @property
    def spec(self) -> FeatureSetSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureSetSpec)

    @property
    def metadata(self) -> VersionedObjMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", VersionedObjMetadata)

    @property
    def status(self) -> FeatureSetStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureSetStatus)

    @property
    def uri(self):
        """fully qualified feature set uri"""
        uri = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        uri = get_store_uri(StorePrefix.FeatureSet, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def get_target_path(self, name=None):
        """get the url/path for an offline or specified data target"""
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def set_targets(self, targets=None, with_defaults=True, default_final_state=None):
        """set the desired target list or defaults

        :param targets:  list of target type names ('csv', 'nosql', ..) or target objects
                         CSVTarget(), ParquetTarget(), NoSqlTarget(), ..
        :param with_defaults: add the default targets (as defined in the central config)
        :param default_final_state: the final graph state after which we add the
                                    target writers, used when the graph branches and
                                    the end cant be determined automatically
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
        if default_final_state:
            self.spec.graph.final_state = default_final_state

    def add_entity(self, entity, name=None):
        """add/set an entity"""
        self._spec.entities.update(entity, name)

    def add_feature(self, feature, name=None):
        """add/set a feature"""
        self._spec.features.update(feature, name)

    def link_analysis(self, name, uri):
        """add a linked file/artifact (chart, data, ..)"""
        self._spec.analysis[name] = uri

    @property
    def graph(self):
        """feature set transformation graph/DAG"""
        return self.spec.graph

    def add_aggregation(
        self,
        name,
        column,
        operations,
        windows,
        period=None,
        state_name=None,
        after=None,
        before=None,
    ):
        """add feature aggregation rule

        example::

            myset.add_aggregation("asks", "ask", ["sum", "max"], ["1h", "5h"], "10m")

        :param name:       aggregation name/prefix
        :param column:     name of column/field aggregate
        :param operations: aggregation operations, e.g. ['sum', 'std']
        :param windows:    list of time windows, e.g. ['1h', '6h', '1d']
        :param period:     optional, sliding window granularity, e.g. '10m'
        :param state_name: optional, graph state name
        :param after:      optional, after which graph state it runs
        :param before:     optional, comes before graph state
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

    def get_stats_table(self):
        """get feature statistics table (as dataframe)"""
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def __getitem__(self, name):
        return self._spec.features[name]

    def __setitem__(self, key, item):
        self._spec.features.update(item, key)

    def plot(self, filename=None, format=None, with_targets=False, **kw):
        """generate graphviz plot"""
        graph = self.spec.graph
        _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)
        targets = None
        if with_targets:
            validate_target_placement(graph, default_final_state, self.spec.targets)
            targets = [
                BaseState(
                    target.kind,
                    after=target.after_state or default_final_state,
                    shape="cylinder",
                )
                for target in self.spec.targets
            ]
        return graph.plot(filename, format, targets=targets, **kw)

    def to_dataframe(self, columns=None, df_module=None, target_name=None):
        """return featureset (offline) data as dataframe"""
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

    def save(self, tag="", versioned=False):
        """save to mlrun db"""
        db = mlrun.get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        as_dict["spec"]["features"] = as_dict["spec"].get(
            "features", []
        )  # bypass DB bug
        db.store_feature_set(as_dict, tag=tag, versioned=versioned)

    def reload(self, update_spec=True):
        """reload/sync the feature vector status and spec from the DB"""
        from_db = mlrun.get_run_db().get_feature_set(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        self.status = from_db.status
        if update_spec:
            self.spec = from_db.spec
