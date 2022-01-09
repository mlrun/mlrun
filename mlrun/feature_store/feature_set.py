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
import datetime
import warnings
from typing import Dict, List, Optional, Union

import pandas as pd

import mlrun
import mlrun.api.schemas

from ..config import config as mlconf
from ..datastore import get_store_uri
from ..datastore.targets import (
    TargetTypes,
    default_target_names,
    get_offline_target,
    get_target_driver,
    validate_target_list,
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
from ..serving.states import BaseStep, RootFlowStep, previous_step
from ..utils import StorePrefix
from .common import verify_feature_set_permissions

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
        output_path=None,
    ):
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._graph: RootFlowStep = None
        self._source = None
        self._engine = None
        self._function: FunctionReference = None

        self.owner = owner
        self.description = description
        self.entities: List[Union[Entity, str]] = entities or []
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
        self.output_path = output_path or mlconf.artifact_path

    @property
    def entities(self) -> List[Entity]:
        """feature set entities (indexes)"""
        return self._entities

    @entities.setter
    def entities(self, entities: List[Union[Entity, str]]):
        if entities:
            # if the entity is a string, convert it to Entity class
            for i, entity in enumerate(entities):
                if isinstance(entity, str):
                    entities[i] = Entity(entity)
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
    def engine(self) -> str:
        """feature set processing engine (storey, pandas, spark)"""
        return self._engine

    @engine.setter
    def engine(self, engine: str):
        engine_list = ["pandas", "spark", "storey"]
        if engine and engine not in engine_list:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"engine must be one of {','.join(engine_list)}"
            )
        self.graph.engine = "sync" if engine and engine in ["pandas", "spark"] else None
        self._engine = engine

    @property
    def graph(self) -> RootFlowStep:
        """feature set transformation graph/DAG"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowStep)
        self._graph.engine = (
            "sync" if self.engine and self.engine in ["pandas", "spark"] else None
        )

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
        return len(self._graph.steps) > 0


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

    def update_last_written_for_target(
        self, target_path: str, last_written: datetime.datetime
    ):
        for target in self._targets:
            if target.path == target_path or target.path.rstrip("/") == target_path:
                target.last_written = last_written


class FeatureSet(ModelObj):
    """Feature set object, defines a set of features and their data pipeline"""

    kind = mlrun.api.schemas.ObjectKind.feature_set.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name: str = None,
        description: str = None,
        entities: List[Union[Entity, str]] = None,
        timestamp_key: str = None,
        engine: str = None,
    ):
        """Feature set object, defines a set of features and their data pipeline

        example::

            import mlrun.feature_store as fstore
            ticks = fstore.FeatureSet("ticks", entities=["stock"], timestamp_key="timestamp")
            fstore.ingest(ticks, df)

        :param name:          name of the feature set
        :param description:   text description
        :param entities:      list of entity (index key) names or :py:class:`~mlrun.features.FeatureSet.Entity`
        :param timestamp_key: timestamp column name
        :param engine:        name of the processing engine (storey, pandas, or spark), defaults to storey
        """
        self._spec: FeatureSetSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None
        self._run_db = None

        self.spec = FeatureSetSpec(
            description=description,
            entities=entities,
            timestamp_key=timestamp_key,
            engine=engine,
        )
        self.metadata = VersionedObjMetadata(name=name)
        self.status = None
        self._last_state = ""
        self._aggregations = {}

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
        return get_store_uri(StorePrefix.FeatureSet, self.fullname)

    @property
    def fullname(self):
        """full name in the form project/name[:tag]"""
        fullname = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        if self._metadata.tag:
            fullname += ":" + self._metadata.tag
        return fullname

    def _override_run_db(
        self, session,
    ):
        # Import here, since this method only runs in API context. If this import was global, client would need
        # API requirements and would fail.
        from ..api.api.utils import get_run_db_instance

        self._run_db = get_run_db_instance(session)

    def _get_run_db(self):
        if self._run_db:
            return self._run_db
        else:
            return mlrun.get_run_db()

    def get_target_path(self, name=None):
        """get the url/path for an offline or specified data target"""
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def set_targets(
        self,
        targets=None,
        with_defaults=True,
        default_final_step=None,
        default_final_state=None,
    ):
        """set the desired target list or defaults

        :param targets:  list of target type names ('csv', 'nosql', ..) or target objects
                         CSVTarget(), ParquetTarget(), NoSqlTarget(), ..
        :param with_defaults: add the default targets (as defined in the central config)
        :param default_final_step: the final graph step after which we add the
                                    target writers, used when the graph branches and
                                    the end cant be determined automatically
        :param default_final_state: *Deprecated* - use default_final_step instead
        """
        if default_final_state:
            warnings.warn(
                "The default_final_state parameter is deprecated. Use default_final_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            default_final_step = default_final_step or default_final_state

        if targets is not None and not isinstance(targets, list):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "targets can only be None or a list of kinds or DataTargetBase derivatives"
            )
        targets = targets or []
        if with_defaults:
            targets.extend(default_target_names())

        validate_target_list(targets=targets)

        for target in targets:
            kind = target.kind if hasattr(target, "kind") else target
            if kind not in TargetTypes.all():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"target kind is not supported, use one of: {','.join(TargetTypes.all())}"
                )
            if not hasattr(target, "kind"):
                target = DataTargetBase(target, name=str(target))
            self.spec.targets.update(target)
        if default_final_step:
            self.spec.graph.final_step = default_final_step

    def purge_targets(self, target_names: List[str] = None, silent: bool = False):
        """ Delete data of specific targets
        :param target_names: List of names of targets to delete (default: delete all ingested targets)
        :param silent: Fail silently if target doesn't exist in featureset status """

        verify_feature_set_permissions(
            self, mlrun.api.schemas.AuthorizationAction.delete
        )

        try:
            self.reload(update_spec=False)
        except mlrun.errors.MLRunNotFoundError:
            # If the feature set doesn't exist in DB there shouldn't be any target to delete
            if silent:
                return
            else:
                raise

        if target_names:
            purge_targets = ObjectList(DataTarget)
            for target_name in target_names:
                try:
                    purge_targets[target_name] = self.status.targets[target_name]
                except KeyError:
                    if silent:
                        pass
                    else:
                        raise mlrun.errors.MLRunNotFoundError(
                            "Target not found in status (fset={0}, target={1})".format(
                                self.name, target_name
                            )
                        )
        else:
            purge_targets = self.status.targets
        purge_target_names = list(purge_targets.keys())
        for target_name in purge_target_names:
            target = purge_targets[target_name]
            driver = get_target_driver(target_spec=target, resource=self)
            try:
                driver.purge()
            except FileNotFoundError:
                pass
            del self.status.targets[target_name]
        self.save()

    def has_valid_source(self):
        """check if object's spec has a valid (non empty) source definition"""
        source = self.spec.source
        return source is not None and source.path is not None and source.path != "None"

    def add_entity(
        self,
        name: str,
        value_type: mlrun.data_types.ValueType = None,
        description: str = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """add/set an entity (dataset index)

        :param name:        entity name
        :param value_type:  type of the entity (default to ValueType.STRING)
        :param description: description of the entity
        :param labels:      label tags dict
        """
        entity = Entity(name, value_type, description=description, labels=labels)
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

    def _add_agregation_to_existing(self, new_aggregation):
        name = new_aggregation["name"]
        if name in self._aggregations:
            current_aggr = self._aggregations[name]
            if current_aggr["windows"] != new_aggregation["windows"]:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Aggregation with name {name} already exists but with window {current_aggr['windows']}. "
                    f"Please provide name for the aggregation"
                )
            if current_aggr["period"] != new_aggregation["period"]:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Aggregation with name {name} already exists but with period {current_aggr['period']}. "
                    f"Please provide name for the aggregation"
                )
            if current_aggr["column"] != new_aggregation["column"]:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Aggregation with name {name} already exists but for different column {current_aggr['column']}. "
                    f"Please provide name for the aggregation"
                )
            current_aggr["operations"] = list(
                set(current_aggr["operations"] + new_aggregation["operations"])
            )

            return
        self._aggregations[name] = new_aggregation

    def add_aggregation(
        self,
        column,
        operations,
        windows,
        period=None,
        name=None,
        step_name=None,
        after=None,
        before=None,
        state_name=None,
    ):
        """add feature aggregation rule

        example::

            myset.add_aggregation("ask", ["sum", "max"], "1h", "10m", name="asks")

        :param column:     name of column/field aggregate
        :param operations: aggregation operations, e.g. ['sum', 'std']
        :param windows:    time windows, can be a single window, e.g. '1h', '1d',
                            or a list of same unit windows e.g ['1h', '6h']
                            windows are transformed to fixed windows or
                            sliding windows depending whether period parameter
                            provided.

                            - Sliding window is fixed-size overlapping windows
                              that slides with time.
                              The window size determines the size of the sliding window
                              and the period determines the step size to slide.
                              Period must be integral divisor of the window size.
                              If the period is not provided then fixed windows is used.

                            - Fixed window is fixed-size, non-overlapping, gap-less window.
                              The window is referred to as a tumbling window.
                              In this case, each record on an in-application stream belongs
                              to a specific window. It is processed only once
                              (when the query processes the window to which the record belongs).
        :param period:     optional, sliding window granularity, e.g. '20s' '10m'  '3h' '7d'
        :param name:       optional, aggregation name/prefix. Must be unique per feature set.If not passed,
                            the column will be used as name.
        :param step_name: optional, graph step name
        :param state_name: *Deprecated* - use step_name instead
        :param after:      optional, after which graph step it runs
        :param before:     optional, comes before graph step
        """
        if isinstance(operations, str):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid parameters provided - operations must be a list."
            )
        if state_name:
            warnings.warn(
                "The state_name parameter is deprecated. Use step_name instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            step_name = step_name or state_name

        name = name or column

        if isinstance(windows, str):
            windows = [windows]
        if isinstance(operations, str):
            operations = [operations]

        aggregation = FeatureAggregation(
            name, column, operations, windows, period
        ).to_dict()

        def upsert_feature(name):
            if name in self.spec.features:
                self.spec.features[name].aggregate = True
            else:
                self.spec.features[name] = Feature(name=column, aggregate=True)

        step_name = step_name or aggregates_step
        graph = self.spec.graph
        if step_name in graph.steps:
            step = graph.steps[step_name]
            self._add_agregation_to_existing(aggregation)
            step.class_args["aggregates"] = list(self._aggregations.values())
        else:
            class_args = {}
            self._aggregations[aggregation["name"]] = aggregation
            step = graph.add_step(
                name=step_name,
                after=after or previous_step,
                before=before,
                class_name="storey.AggregateByKey",
                aggregates=[aggregation],
                table=".",
                **class_args,
            )

        for operation in operations:
            for window in windows:
                upsert_feature(f"{name}_{operation}_{window}")

        return step

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
        _, default_final_step, _ = graph.check_and_process_graph(allow_empty=True)
        targets = None
        if with_targets:
            validate_target_list(targets=targets)
            validate_target_placement(graph, default_final_step, self.spec.targets)
            targets = [
                BaseStep(
                    target.kind,
                    after=target.after_step or default_final_step,
                    shape="cylinder",
                )
                for target in self.spec.targets
            ]
        return graph.plot(filename, format, targets=targets, **kw)

    def to_dataframe(
        self,
        columns=None,
        df_module=None,
        target_name=None,
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        """return featureset (offline) data as dataframe

        :param columns:      list of columns to select (if not all)
        :param df_module:    py module used to create the DataFrame (pd for Pandas, dd for Dask, ..)
        :param target_name:  select a specific target (material view)
        :param start_time:   filter by start time
        :param end_time:     filter by end time
        :param time_column:  specify the time column name in the file
        :param kwargs:       additional reader (csv, parquet, ..) args
        :return: DataFrame
        """
        entities = list(self.spec.entities.keys())
        if columns:
            if self.spec.timestamp_key and self.spec.timestamp_key not in entities:
                columns = [self.spec.timestamp_key] + columns
            columns = entities + columns
        driver = get_offline_target(self, name=target_name)
        if not driver:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature set"
            )
        return driver.as_df(
            columns=columns,
            df_module=df_module,
            entities=entities,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            **kwargs,
        )

    def save(self, tag="", versioned=False):
        """save to mlrun db"""
        db = self._get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag or "latest"
        as_dict = self.to_dict()
        as_dict["spec"]["features"] = as_dict["spec"].get(
            "features", []
        )  # bypass DB bug
        db.store_feature_set(as_dict, tag=tag, versioned=versioned)

    def reload(self, update_spec=True):
        """reload/sync the feature vector status and spec from the DB"""
        feature_set = self._get_run_db().get_feature_set(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        if isinstance(feature_set, dict):
            feature_set = FeatureSet.from_dict(feature_set)

        self.status = feature_set.status
        if update_spec:
            self.spec = feature_set.spec
