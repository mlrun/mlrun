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
import warnings
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from storey import EmitEveryEvent, EmitPolicy

import mlrun
import mlrun.common.schemas

from ..config import config as mlconf
from ..data_types import InferOptions
from ..datastore import get_store_uri
from ..datastore.sources import BaseSourceDriver, source_kind_to_driver
from ..datastore.targets import (
    TargetTypes,
    get_default_targets,
    get_offline_target,
    get_online_target,
    get_target_driver,
    update_targets_run_id_for_ingest,
    validate_target_list,
    validate_target_placement,
)
from ..features import Entity, Feature
from ..model import (
    DataSource,
    DataTarget,
    DataTargetBase,
    ModelObj,
    ObjectDict,
    ObjectList,
    VersionedObjMetadata,
)
from ..runtimes import BaseRuntime
from ..runtimes.function_reference import FunctionReference
from ..serving.states import BaseStep, RootFlowStep, previous_step, queue_class_names
from ..serving.utils import StepToDict
from ..utils import StorePrefix, logger
from .common import RunConfig, verify_feature_set_permissions

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
        passthrough=None,
    ):
        """Feature set spec object, defines the feature-set's configuration.

        .. warning::
            This class should not be modified directly. It is managed by the parent feature-set object or using
            feature-store APIs. Modifying the spec manually may result in unpredictable behaviour.

        :param description:   text description (copied from parent feature-set)
        :param entities:      list of entity (index key) names or :py:class:`~mlrun.features.FeatureSet.Entity`
        :param features: list of features - :py:class:`~mlrun.features.FeatureSet.Feature`
        :param partition_keys: list of fields to partition results by (other than the default timestamp key)
        :param timestamp_key: timestamp column name
        :param label_column: name of the label column (the one holding the target (y) values)
        :param targets: list of data targets
        :param graph: the processing graph
        :param function: MLRun runtime to execute the feature-set in
        :param engine: name of the processing engine (storey, pandas, or spark), defaults to storey
        :param output_path: default location where to store results (defaults to MLRun's artifact path)
        :param passthrough: if true, ingest will skip offline targets, and get_offline_features will
               read directly from source
        """
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._graph: RootFlowStep = None
        self._source = None
        self._engine = None
        self._function: FunctionReference = None
        self._relations: ObjectDict = None

        self.owner = owner
        self.description = description
        self.entities: list[Union[Entity, str]] = entities or []
        self.relations: dict[str, Union[Entity, str]] = relations or {}
        self.features: list[Feature] = features or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.source = source
        self.targets = targets or []
        self.graph = graph
        self.label_column = label_column
        self.function = function
        self.analysis = analysis or {}
        self.engine = engine
        self.output_path = output_path or mlconf.artifact_path
        self.passthrough = passthrough
        self.with_default_targets = True

    @property
    def entities(self) -> list[Entity]:
        """feature set entities (indexes)"""
        return self._entities

    @entities.setter
    def entities(self, entities: list[Union[Entity, str]]):
        if entities:
            # if the entity is a string, convert it to Entity class
            for i, entity in enumerate(entities):
                if isinstance(entity, str):
                    entities[i] = Entity(entity)
                elif isinstance(entity, Entity) and entity.name is None:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "You have to provide an "
                        "Entity with valid name of string type"
                    )
                elif isinstance(entity, dict) and (
                    "name" not in entity
                    or ("name" in entity and entity["name"] is None)
                ):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "You have to provide an "
                        "Entity with valid name of string type"
                    )
        self._entities = ObjectList.from_list(Entity, entities)

    @property
    def features(self) -> list[Feature]:
        """feature set features list"""
        return self._features

    @features.setter
    def features(self, features: list[Feature]):
        self._features = ObjectList.from_list(Feature, features)

    @property
    def targets(self) -> list[DataTargetBase]:
        """list of desired targets (material storage)"""
        return self._targets

    @targets.setter
    def targets(self, targets: list[DataTargetBase]):
        self._targets = ObjectList.from_list(DataTargetBase, targets)

    @property
    def engine(self) -> str:
        """feature set processing engine (storey, pandas, spark)"""
        return self._engine

    @engine.setter
    def engine(self, engine: str):
        engine_list = ["pandas", "spark", "storey"]
        engine = engine if engine else "storey"
        if engine not in engine_list:
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
    def source(self, source: Union[BaseSourceDriver, dict]):
        if isinstance(source, dict):
            kind = source.get("kind", "")
            source = source_kind_to_driver[kind].from_dict(source)
        self._source = source

    @property
    def relations(self) -> dict[str, Entity]:
        """feature set relations dict"""
        return self._relations

    @relations.setter
    def relations(self, relations: dict[str, Entity]):
        for col, ent in relations.items():
            if isinstance(ent, str):
                relations[col] = Entity(ent)
        self._relations = ObjectDict.from_dict({"entity": Entity}, relations, "entity")

    def require_processing(self):
        return len(self._graph.steps) > 0

    def validate_no_processing_for_passthrough(self):
        if self.passthrough and self.require_processing():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "passthrough feature set can not have graph transformations"
            )


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
        """Feature set status object, containing the current feature-set's status.

        .. warning::
            This class should not be modified directly. It is managed by the parent feature-set object or using
            feature-store APIs. Modifying the status manually may result in unpredictable behaviour.

        :param state: object's current state
        :param targets: list of the data targets used in the last ingestion operation
        :param stats: feature statistics calculated in the last ingestion (if stats calculation was requested)
        :param preview: preview of the feature-set contents (if preview generation was requested)
        :param function_uri: function used to execute the feature-set graph
        :param run_uri: last run used for ingestion
        """

        self.state = state or mlrun.common.schemas.object.ObjectStatusState.CREATED
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = preview or []
        self.function_uri = function_uri
        self.run_uri = run_uri

    @property
    def targets(self) -> list[DataTarget]:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: list[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    def update_last_written_for_target(self, target_path: str, last_written: datetime):
        for target in self._targets:
            actual_target_path = get_target_driver(target).get_target_path()
            if (
                actual_target_path == target_path
                or actual_target_path.rstrip("/") == target_path
            ):
                target.last_written = last_written


def emit_policy_to_dict(policy: EmitPolicy):
    # Storey expects the policy to be converted to a dictionary with specific params and won't allow extra params
    # (see Storey's _dict_to_emit_policy function). This takes care of creating a dict conforming to it.
    # TODO - fix Storey's handling of emit policy and parsing of dict in _dict_to_emit_policy.
    struct = {"mode": policy.name()}
    if hasattr(policy, "delay_in_seconds"):
        struct["delay"] = getattr(policy, "delay_in_seconds")
    if hasattr(policy, "max_events"):
        struct["maxEvents"] = getattr(policy, "max_events")
    return struct


class FeatureSet(ModelObj):
    kind = mlrun.common.schemas.ObjectKind.feature_set.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        entities: Optional[list[Union[Entity, str]]] = None,
        timestamp_key: Optional[str] = None,
        engine: Optional[str] = None,
        label_column: Optional[str] = None,
        relations: Optional[dict[str, Union[Entity, str]]] = None,
        passthrough: Optional[bool] = None,
    ):
        """Feature set object, defines a set of features and their data pipeline

        example::

            import mlrun.feature_store as fstore

            ticks = fstore.FeatureSet(
                "ticks", entities=["stock"], timestamp_key="timestamp"
            )
            ticks.ingest(df)

        :param name:          name of the feature set
        :param description:   text description
        :param entities:      list of entity (index key) names or :py:class:`~mlrun.features.FeatureSet.Entity`
        :param timestamp_key: timestamp column name
        :param engine:        name of the processing engine (storey, pandas, or spark), defaults to storey
        :param label_column:  name of the label column (the one holding the target (y) values)
        :param relations:     dictionary that indicates all the relations this feature set
                              have with another feature sets. The format of this dictionary is
                              {"my_column":Entity, ...}
        :param passthrough:   if true, ingest will skip offline targets, and get_offline_features will read
                              directly from source
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
            label_column=label_column,
            relations=relations,
            passthrough=passthrough,
        )

        if timestamp_key in self.spec.entities.keys():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "timestamp key can not be entity"
            )

        self.metadata = VersionedObjMetadata(name=name)
        self.status = None
        self._last_state = ""
        self._aggregations = {}
        self.set_targets()

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
    def fullname(self) -> str:
        """full name in the form ``{project}/{name}[:{tag}]``"""
        fullname = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        if self._metadata.tag:
            fullname += ":" + self._metadata.tag
        return fullname

    def _get_run_db(self):
        if self._run_db:
            return self._run_db
        else:
            return mlrun.get_run_db()

    def _override_run_db(self, run_db):
        self._run_db = run_db

    def get_target_path(self, name=None):
        """get the url/path for an offline or specified data target"""
        target = get_offline_target(self, name=name)

        if not target and name:
            target = get_online_target(self, name)

        if target:
            return target.get_path().get_absolute_path(
                project_name=self.metadata.project
            )

    def set_targets(
        self,
        targets=None,
        with_defaults=True,
        default_final_step=None,
        default_final_state=None,
    ):
        """set the desired target list or defaults

        :param targets:  list of target type names ('csv', 'nosql', ..) or target objects
                         CSVTarget(), ParquetTarget(), NoSqlTarget(), StreamTarget(), ..
        :param with_defaults: add the default targets (as defined in the central config)
        :param default_final_step: the final graph step after which we add the
                                    target writers, used when the graph branches and
                                    the end cant be determined automatically
        :param default_final_state: *Deprecated* - use default_final_step instead
        """
        if default_final_state:
            warnings.warn(
                "The 'default_final_state' parameter is deprecated in 1.3.0 and will be remove in 1.5.0. "
                "Use 'default_final_step' instead.",
                # TODO: remove in 1.5.0
                FutureWarning,
            )
            default_final_step = default_final_step or default_final_state

        if targets is not None and not isinstance(targets, list):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "targets can only be None or a list of kinds or DataTargetBase derivatives"
            )
        targets = targets or []
        if with_defaults:
            self.spec.with_default_targets = True
            targets.extend(get_default_targets(offline_only=self.spec.passthrough))
        else:
            self.spec.with_default_targets = False

        self.spec.targets = []
        self.__set_targets_add_targets_helper(targets)

        if default_final_step:
            self.spec.graph.final_step = default_final_step

    def __set_targets_add_targets_helper(self, targets):
        """
        Add the desired target list

        :param targets: list of target type names ('csv', 'nosql', ..) or target objects
                         CSVTarget(), ParquetTarget(), NoSqlTarget(), StreamTarget(), ..
        """
        validate_target_list(targets=targets)
        for target in targets:
            kind = target.kind if hasattr(target, "kind") else target
            if kind not in TargetTypes.all():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"target kind is not supported, use one of: {','.join(TargetTypes.all())}"
                )
            if not hasattr(target, "kind"):
                target = DataTargetBase(
                    target, name=str(target), partitioned=(target == "parquet")
                )
            self.spec.targets.update(target)

    def validate_steps(self, namespace):
        if not self.spec:
            return
        if not self.spec.graph:
            return
        for step in self.spec.graph.steps.values():
            if (
                step.class_name in queue_class_names
                or step.class_name is None
                or "." not in step.class_name
            ):
                #  we are not checking none class names or queue class names.
                continue
            class_object, class_name = step.get_step_class_object(namespace=namespace)
            if not hasattr(class_object, "validate_args"):
                continue
            class_args = step.get_full_class_args(
                namespace=namespace, class_object=class_object
            )
            if class_name.startswith("storey"):
                class_object.validate_args(
                    **(class_args if class_args is not None else {})
                )
            else:
                class_object.validate_args(
                    self, **(class_args if class_args is not None else {})
                )

    def purge_targets(
        self, target_names: Optional[list[str]] = None, silent: bool = False
    ):
        """Delete data of specific targets
        :param target_names: List of names of targets to delete (default: delete all ingested targets)
        :param silent: Fail silently if target doesn't exist in featureset status"""

        verify_feature_set_permissions(
            self, mlrun.common.schemas.AuthorizationAction.delete
        )

        purge_targets = self._reload_and_get_status_targets(
            target_names=target_names, silent=silent
        )

        if purge_targets:
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

    def update_targets_for_ingest(
        self,
        targets: list[DataTargetBase],
        overwrite: Optional[bool] = None,
    ):
        if not targets:
            return

        ingestion_target_names = [t.name for t in targets]

        status_targets = {}
        if not overwrite:
            # silent=True always because targets are not guaranteed to be found in status
            status_targets = (
                self._reload_and_get_status_targets(
                    target_names=ingestion_target_names, silent=True
                )
                or {}
            )

        update_targets_run_id_for_ingest(overwrite, targets, status_targets)

    def _reload_and_get_status_targets(
        self, target_names: Optional[list[str]] = None, silent: bool = False
    ):
        try:
            self.reload(update_spec=False)
        except mlrun.errors.MLRunNotFoundError:
            # If the feature set doesn't exist in DB there shouldn't be any target to delete
            if silent:
                return
            else:
                raise

        if target_names:
            targets = ObjectList(DataTarget)
            for target_name in target_names:
                try:
                    targets[target_name] = self.status.targets[target_name]
                except KeyError:
                    if silent:
                        pass
                    else:
                        raise mlrun.errors.MLRunNotFoundError(
                            f"Target not found in status (fset={self.metadata.name}, target={target_name})"
                        )
        else:
            targets = self.status.targets

        return targets

    def has_valid_source(self):
        """check if object's spec has a valid (non empty) source definition"""
        source = self.spec.source
        return source is not None and source.path is not None and source.path != "None"

    def add_entity(
        self,
        name: str,
        value_type: mlrun.data_types.ValueType = None,
        description: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ):
        """add/set an entity (dataset index)

        example::

            import mlrun.feature_store as fstore

            ticks = fstore.FeatureSet(
                "ticks", entities=["stock"], timestamp_key="timestamp"
            )
            ticks.add_entity(
                "country", mlrun.data_types.ValueType.STRING, description="stock country"
            )
            ticks.add_entity("year", mlrun.data_types.ValueType.INT16)
            ticks.save()

        :param name:        entity name
        :param value_type:  type of the entity (default to ValueType.STRING)
        :param description: description of the entity
        :param labels:      label tags dict
        """
        entity = Entity(name, value_type, description=description, labels=labels)
        self._spec.entities.update(entity, name)

    def add_feature(self, feature: mlrun.features.Feature, name=None):
        """add/set a feature

        example::

            import mlrun.feature_store as fstore
            from mlrun.features import Feature

            ticks = fstore.FeatureSet(
                "ticks", entities=["stock"], timestamp_key="timestamp"
            )
            ticks.add_feature(
                Feature(
                    value_type=mlrun.data_types.ValueType.STRING,
                    description="client consistency",
                ),
                "ABC01",
            )
            ticks.add_feature(
                Feature(
                    value_type=mlrun.data_types.ValueType.FLOAT,
                    description="client volatility",
                ),
                "SAB",
            )
            ticks.save()

        :param feature:         setting of Feature
        :param name:            feature name
        """
        self._spec.features.update(feature, name)

    def link_analysis(self, name, uri):
        """add a linked file/artifact (chart, data, ..)"""
        self._spec.analysis[name] = uri

    @property
    def graph(self) -> RootFlowStep:
        """feature set transformation graph/DAG"""
        return self.spec.graph

    def _add_aggregation_to_existing(self, new_aggregation):
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
        emit_policy: EmitPolicy = None,
    ):
        """add feature aggregation rule

        example::

            myset.add_aggregation("ask", ["sum", "max"], "1h", "10m", name="asks")

        :param column:     name of column/field aggregate. Do not name columns starting with either `_` or `aggr_`.
                           They are reserved for internal use, and the data does not ingest correctly.
                           When using the pandas engine, do not use spaces (` `) or periods (`.`) in the column names;
                           they cause errors in the ingestion.
        :param operations: aggregation operations. Supported operations:
                             count, sum, sqr, max, min, first, last, avg, stdvar, stddev
        :param windows:    time windows, can be a single window, e.g. '1h', '1d',
                            or a list of same unit windows e.g. ['1h', '6h']
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
        :param name:       optional, aggregation name/prefix. Must be unique per feature set. If not passed,
                            the column will be used as name.
        :param step_name: optional, graph step name
        :param after:      optional, after which graph step it runs
        :param before:     optional, comes before graph step
        :param emit_policy: optional, which emit policy to use when performing the aggregations. Use the derived
                            classes of ``storey.EmitPolicy``. The default is to emit every period for Spark engine
                            and emit every event for storey. Currently the only other supported option is to use
                            ``emit_policy=storey.EmitEveryEvent()`` when using the Spark engine to emit every event

        """
        if isinstance(operations, str):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invalid parameters provided - operations must be a list."
            )

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
                self.spec.features[name] = Feature(
                    name=column, aggregate=True, value_type="float"
                )

        step_name = step_name or aggregates_step
        graph = self.spec.graph
        if step_name in graph.steps:
            step = graph.steps[step_name]
            self._add_aggregation_to_existing(aggregation)
            step.class_args["aggregates"] = list(self._aggregations.values())
            if emit_policy and self.spec.engine == "spark":
                # Using simple override here - we might want to consider exploding if different emit policies
                # were used for multiple aggregations.
                emit_policy_dict = emit_policy_to_dict(emit_policy)
                if "emit_policy" in step.class_args:
                    curr_emit_policy = step.class_args["emit_policy"]["mode"]
                    if curr_emit_policy != emit_policy_dict["mode"]:
                        logger.warning(
                            f"Current emit policy will be overridden: {curr_emit_policy} => {emit_policy_dict['mode']}"
                        )
                step.class_args["emit_policy"] = emit_policy_dict
        else:
            class_args = {}
            self._aggregations[aggregation["name"]] = aggregation
            if before is None and after is None:
                after = previous_step
            if not self.spec.engine or self.spec.engine == "storey":
                step = graph.add_step(
                    name=step_name,
                    after=after,
                    before=before,
                    class_name="storey.AggregateByKey",
                    time_field=self.spec.timestamp_key,
                    aggregates=[aggregation],
                    table=".",
                    **class_args,
                )
            elif self.spec.engine == "spark":
                key_columns = []
                if emit_policy:
                    class_args["emit_policy"] = emit_policy_to_dict(emit_policy)
                for entity in self.spec.entities:
                    key_columns.append(entity.name)
                step = graph.add_step(
                    name=step_name,
                    key_columns=key_columns,
                    time_column=self.spec.timestamp_key,
                    aggregates=[aggregation],
                    after=after,
                    before=before,
                    class_name="mlrun.feature_store.feature_set.SparkAggregateByKey",
                    **class_args,
                )
            else:
                raise ValueError(
                    "Aggregations are only implemented for storey and spark engines."
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
        if key not in self._spec.entities.keys():
            self._spec.features.update(item, key)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "A `FeatureSet` cannot have an entity and a feature with the same name. "
                f"The feature that was given to add '{key}' has the same name of the `FeatureSet`'s entity."
            )

    def plot(self, filename=None, format=None, with_targets=False, **kw):
        """plot/save graph using graphviz

        example::

            import mlrun.feature_store as fstore

            ...
            ticks = fstore.FeatureSet(
                "ticks", entities=["stock"], timestamp_key="timestamp"
            )
            ticks.add_aggregation(
                name="priceN",
                column="price",
                operations=["avg"],
                windows=["1d"],
                period="1h",
            )
            ticks.plot(rankdir="LR", with_targets=True)

        :param filename:     target filepath for the graph image (None for the notebook)
        :param format:       the output format used for rendering (``'pdf'``, ``'png'``, etc.)
        :param with_targets: show targets in the graph image
        :param kw:           kwargs passed to graphviz, e.g. rankdir=”LR” (see https://graphviz.org/doc/info/attrs.html)
        :return:             graphviz graph object
        """
        graph = self.spec.graph
        _, default_final_step, _ = graph.check_and_process_graph(allow_empty=True)
        targets = None
        if with_targets:
            validate_target_list(targets=targets)
            validate_target_placement(graph, default_final_step, self.spec.targets)
            targets = [
                BaseStep(
                    f"{target.kind}/{target.name}",
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
        additional_filters=None,
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
        :param additional_filters: List of additional_filter conditions as tuples.
                                    Each tuple should be in the format (column_name, operator, value).
                                    Supported operators: "=", ">=", "<=", ">", "<".
                                    Example: [("Product", "=", "Computer")]
                                    For all supported filters, please see:
                                    https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
        :return: DataFrame
        """
        entities = list(self.spec.entities.keys())
        if columns:
            if self.spec.timestamp_key and self.spec.timestamp_key not in entities:
                columns = [self.spec.timestamp_key] + columns
            columns = entities + columns

        if self.spec.passthrough:
            if not self.spec.source:
                raise mlrun.errors.MLRunNotFoundError(
                    "passthrough feature set {self.metadata.name} with no source"
                )
            df = self.spec.source.to_dataframe(
                columns=columns,
                start_time=start_time,
                end_time=end_time,
                time_field=time_column,
                additional_filters=additional_filters,
                **kwargs,
            )
            # to_dataframe() can sometimes return an iterator of dataframes instead of one dataframe
            if not isinstance(df, pd.DataFrame):
                df = pd.concat(df)
            return df

        target = get_offline_target(self, name=target_name)
        if not target:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature set"
            )
        result = target.as_df(
            columns=columns,
            df_module=df_module,
            entities=entities,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            additional_filters=additional_filters,
            **kwargs,
        )
        return result

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

    def ingest(
        self,
        source=None,
        targets: Optional[list[DataTargetBase]] = None,
        namespace=None,
        return_df: bool = True,
        infer_options: InferOptions = InferOptions.default(),
        run_config: RunConfig = None,
        mlrun_context=None,
        spark_context=None,
        overwrite=None,
    ) -> Optional[pd.DataFrame]:
        """Read local DataFrame, file, URL, or source into the feature store
        Ingest reads from the source, run the graph transformations, infers  metadata and stats
        and writes the results to the default of specified targets

        when targets are not specified data is stored in the configured default targets
        (will usually be NoSQL for real-time and Parquet for offline).

        the `run_config` parameter allow specifying the function and job configuration,
        see: :py:class:`~mlrun.feature_store.RunConfig`

        example::

            stocks_set = FeatureSet("stocks", entities=[Entity("ticker")])
            stocks = pd.read_csv("stocks.csv")
            df = stocks_set.ingest(stocks, infer_options=fstore.InferOptions.default())

            # for running as remote job
            config = RunConfig(image="mlrun/mlrun")
            df = ingest(stocks_set, stocks, run_config=config)

            # specify source and targets
            source = CSVSource("mycsv", path="measurements.csv")
            targets = [CSVTarget("mycsv", path="./mycsv.csv")]
            ingest(measurements, source, targets)

        :param source:        source dataframe or other sources (e.g. parquet source see:
                            :py:class:`~mlrun.datastore.ParquetSource` and other classes in mlrun.datastore with suffix
                            Source)
        :param targets:       optional list of data target objects
        :param namespace:     namespace or module containing graph classes
        :param return_df:     indicate if to return a dataframe with the graph results
        :param infer_options: schema (for discovery of entities, features in featureset), index, stats,
                            histogram and preview infer options (:py:class:`~mlrun.feature_store.InferOptions`)
        :param run_config:    function and/or run configuration for remote jobs,
                            see :py:class:`~mlrun.feature_store.RunConfig`
        :param mlrun_context: mlrun context (when running as a job), for internal use !
        :param spark_context: local spark session for spark ingestion, example for creating the spark context:
                            `spark = SparkSession.builder.appName("Spark function").getOrCreate()`
                            For remote spark ingestion, this should contain the remote spark service name
        :param overwrite:     delete the targets' data prior to ingestion
                            (default: True for non scheduled ingest - deletes the targets that are about to be ingested.
                            False for scheduled ingest - does not delete the target)
        :return:              if return_df is True, a dataframe will be returned based on the graph
        """
        return mlrun.feature_store.api._ingest(
            self,
            source,
            targets,
            namespace,
            return_df,
            infer_options,
            run_config,
            mlrun_context,
            spark_context,
            overwrite,
        )

    def preview(
        self,
        source,
        entity_columns: Optional[list] = None,
        namespace=None,
        options: InferOptions = None,
        verbose: bool = False,
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """run the ingestion pipeline with local DataFrame/file data and infer features schema and stats

        example::

            quotes_set = FeatureSet("stock-quotes", entities=[Entity("ticker")])
            quotes_set.add_aggregation("ask", ["sum", "max"], ["1h", "5h"], "10m")
            quotes_set.add_aggregation("bid", ["min", "max"], ["1h"], "10m")
            df = quotes_set.preview(
                quotes_df,
                entity_columns=["ticker"],
            )

        :param source:         source dataframe or csv/parquet file path
        :param entity_columns: list of entity (index) column names
        :param namespace:      namespace or module containing graph classes
        :param options:        schema (for discovery of entities, features in featureset), index, stats,
                            histogram and preview infer options (:py:class:`~mlrun.feature_store.InferOptions`)
        :param verbose:        verbose log
        :param sample_size:    num of rows to sample from the dataset (for large datasets)
        """
        return mlrun.feature_store.api._preview(
            self, source, entity_columns, namespace, options, verbose, sample_size
        )

    def deploy_ingestion_service(
        self,
        source: DataSource = None,
        targets: Optional[list[DataTargetBase]] = None,
        name: Optional[str] = None,
        run_config: RunConfig = None,
        verbose=False,
    ) -> tuple[str, BaseRuntime]:
        """Start real-time ingestion service using nuclio function

        Deploy a real-time function implementing feature ingestion pipeline
        the source maps to Nuclio event triggers (http, kafka, v3io stream, etc.)

        the `run_config` parameter allow specifying the function and job configuration,
        see: :py:class:`~mlrun.feature_store.RunConfig`

        example::

            source = HTTPSource()
            func = mlrun.code_to_function("ingest", kind="serving").apply(mount_v3io())
            config = RunConfig(function=func)
            my_set.deploy_ingestion_service(source, run_config=config)

        :param source:        data source object describing the online or offline source
        :param targets:       list of data target objects
        :param name:          name for the job/function
        :param run_config:    service runtime configuration (function object/uri, resources, etc..)
        :param verbose:       verbose log

        :return: URL to access the deployed ingestion service, and the function that was deployed (which will
                differ from the function passed in via the run_config parameter).
        """

        return mlrun.feature_store.api._deploy_ingestion_service_v2(
            self, source, targets, name, run_config, verbose
        )

    def extract_relation_keys(
        self,
        other_feature_set,
        relations: Optional[dict[str, Union[str, Entity]]] = None,
    ) -> list[str]:
        """
        Checks whether a feature set can be merged to the right of this feature set.

        :param other_feature_set:   The feature set to be merged to the right of this feature set.
        :param relations:           The relations that were defined on this feature set.
        :returns:                   If the two feature sets can be merged, a list of the left join keys is returned.
                                    Otherwise, an empty list is returned.
                                    (The right join keys are always the entities of the other feature set)
        """
        right_feature_set_entity_list = other_feature_set.spec.entities

        if all(
            ent in self.spec.entities for ent in right_feature_set_entity_list
        ) and len(right_feature_set_entity_list) == len(self.spec.entities):
            # entities wise
            return list(self.spec.entities.keys())
        elif all(ent in self.spec.entities for ent in right_feature_set_entity_list):
            # entities wise when the right fset have lower number of entities
            return list(right_feature_set_entity_list.keys())

        elif relations:
            curr_col_relations_list = list(
                map(
                    lambda ent: (
                        list(relations.keys())[list(relations.values()).index(ent)]
                        if ent in relations.values()
                        else False
                    ),
                    right_feature_set_entity_list,
                )
            )

            if all(curr_col_relations_list):
                return curr_col_relations_list

        return []

    def is_connectable_to_df(self, df_columns: list[str]) -> bool:
        """
        This method checks if the dataframe can be left-joined with this feature set

        :param df_columns:  The columns of the data frame you want to merge to the left of this feature set
        :return:            True if it can be left-joined and False otherwise
        """
        return df_columns and all(
            ent in df_columns for ent in self.spec.entities.keys()
        )


class SparkAggregateByKey(StepToDict):
    _supported_operations = [
        "count",
        "sum",
        "sqr",
        "max",
        "min",
        "first",
        "last",
        "avg",
        "stdvar",
        "stddev",
    ]

    def __init__(
        self,
        key_columns: list[str],
        time_column: str,
        aggregates: list[dict],
        emit_policy: Union[EmitPolicy, dict] = None,
    ):
        self.key_columns = key_columns
        self.time_column = time_column
        self.aggregates = aggregates
        self.emit_policy_mode = None
        if emit_policy:
            if isinstance(emit_policy, EmitPolicy):
                emit_policy = emit_policy_to_dict(emit_policy)
            self.emit_policy_mode = emit_policy["mode"]

    @staticmethod
    def _duration_to_spark_format(duration):
        num = duration[:-1]
        unit = duration[-1:]
        if unit == "d":
            unit = "day"
        elif unit == "h":
            unit = "hour"
        elif unit == "m":
            unit = "minute"
        elif unit == "s":
            unit = "second"
        else:
            raise ValueError(f"Invalid duration '{duration}'")
        return f"{num} {unit}"

    @staticmethod
    def _verify_operation(op):
        if op not in SparkAggregateByKey._supported_operations:
            error_string = (
                f"operation {op} is unsupported. Supported operations: "
                + ", ".join(SparkAggregateByKey._supported_operations)
            )
            raise mlrun.errors.MLRunInvalidArgumentError(error_string)

    def _extract_fields_from_aggregate_dict(self, aggregate):
        name = aggregate["name"]
        column = aggregate["column"]
        operations = aggregate["operations"]
        for op in operations:
            self._verify_operation(op)
        windows = aggregate["windows"]
        spark_period = (
            self._duration_to_spark_format(aggregate["period"])
            if "period" in aggregate
            else None
        )
        return name, column, operations, windows, spark_period

    @staticmethod
    def _get_aggr(operation, column):
        import pyspark.sql.functions as funcs

        if operation == "sqr":
            return funcs.sum(funcs.expr(f"{column} * {column}"))
        elif operation == "stdvar":
            return funcs.variance(column)
        else:
            func = getattr(funcs, operation)
            return func(column)

    def do(self, event):
        import pyspark.sql.functions as funcs
        from pyspark.sql import Window

        time_column = self.time_column or "time"
        input_df = event

        if not self.emit_policy_mode or self.emit_policy_mode != EmitEveryEvent.name():
            last_value_aggs = [
                funcs.last(column).alias(column)
                for column in input_df.columns
                if column not in self.key_columns and column != time_column
            ]

            dfs = []
            for aggregate in self.aggregates:
                (
                    name,
                    column,
                    operations,
                    windows,
                    spark_period,
                ) = self._extract_fields_from_aggregate_dict(aggregate)

                for window in windows:
                    spark_window = self._duration_to_spark_format(window)
                    aggs = last_value_aggs
                    for operation in operations:
                        agg = self._get_aggr(operation, column)
                        agg_name = f"{name if name else column}_{operation}_{window}"
                        agg = agg.alias(agg_name)
                        aggs.append(agg)
                    window_column = funcs.window(
                        time_column, spark_window, spark_period
                    )
                    df = input_df.groupBy(
                        *self.key_columns,
                        window_column.end.alias(time_column),
                    ).agg(*aggs)
                    df = df.withColumn(f"{time_column}_window", funcs.lit(window))
                    dfs.append(df)

            union_df = dfs[0]
            for df in dfs[1:]:
                union_df = union_df.unionByName(df, allowMissingColumns=True)

            return union_df

        else:
            window_counter = 0
            # We'll use this column to identify our original row and group-by across the various windows
            # (either sliding windows or multiple windows provided). See below comment for more details.
            rowid_col = "__mlrun_rowid"
            df = input_df.withColumn(rowid_col, funcs.monotonically_increasing_id())

            drop_columns = [rowid_col]
            window_rank_cols = []
            union_df = None
            for aggregate in self.aggregates:
                (
                    name,
                    column,
                    operations,
                    windows,
                    spark_period,
                ) = self._extract_fields_from_aggregate_dict(aggregate)

                for window in windows:
                    spark_window = self._duration_to_spark_format(window)
                    window_col = f"__mlrun_window_{window_counter}"
                    win_df = df.withColumn(
                        window_col,
                        funcs.window(time_column, spark_window, spark_period).end,
                    )
                    function_window = Window.partitionBy(*self.key_columns, window_col)

                    window_rank_col = f"__mlrun_win_rank_{window_counter}"
                    rank_window = Window.partitionBy(rowid_col).orderBy(window_col)
                    win_df = win_df.withColumn(
                        window_rank_col, funcs.row_number().over(rank_window)
                    )
                    window_rank_cols.append(window_rank_col)
                    drop_columns.extend([window_col, window_rank_col])

                    window_counter += 1

                    for operation in operations:
                        agg = self._get_aggr(operation, column)
                        agg_name = f"{name if name else column}_{operation}_{window}"
                        win_df = win_df.withColumn(agg_name, agg.over(function_window))

                    union_df = (
                        union_df.unionByName(win_df, allowMissingColumns=True)
                        if union_df
                        else win_df
                    )

            # We need to collapse the multiple window rows that were generated during the query processing. For that
            # purpose we'll pick just the 1st row for each window, and then group-by with ignorenulls. Basically since
            # the result is a union of multiple windows, we'll get something like this for each input row:
            # row   window_1    rank_window_1   window_2    rank_window_2   ...calculations and fields...
            # 1     10:00       1               null        null            ...
            # 2     10:10       2               null        null            ...
            # 3     null        null            10:00       1               ...
            # 4     null        null            10:10       2               ...
            # And we want to take rows 1 and 3 in this case. Then the group-by will merge them to a single line since
            # it ignores nulls, so it will take the values for window_1 from row 1 and for window_2 from row 3.
            window_filter = " or ".join(
                [f"{window_rank_col} == 1" for window_rank_col in window_rank_cols]
            )
            first_value_aggs = [
                funcs.first(column, ignorenulls=True).alias(column)
                for column in union_df.columns
                if column not in drop_columns
            ]

            return (
                union_df.filter(window_filter)
                .groupBy(rowid_col)
                .agg(*first_value_aggs)
                .drop(rowid_col)
            )
