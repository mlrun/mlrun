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
import collections
import logging
import typing
from copy import copy
from datetime import datetime
from enum import Enum
from typing import Union

import pandas as pd

import mlrun

from ..config import config as mlconf
from ..datastore import get_store_uri
from ..datastore.targets import BaseStoreTarget, get_offline_target
from ..feature_store.common import (
    get_feature_set_by_uri,
    parse_feature_string,
    parse_project_name_from_feature_string,
    verify_feature_vector_permissions,
)
from ..feature_store.feature_set import FeatureSet
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
from ..runtimes.function_reference import FunctionReference
from ..serving.states import RootFlowStep
from ..utils import StorePrefix
from .common import RunConfig
from .feature_vector_utils import JoinGraph, OnlineVectorService
from .retrieval import get_merger, run_merge_job


def _features_to_vector_and_check_permissions(features: "FeatureVector", update_stats):
    vector = features
    if not vector.metadata.name:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "feature vector name must be specified"
        )
    verify_feature_vector_permissions(
        vector, mlrun.common.schemas.AuthorizationAction.update
    )

    vector.save()
    return vector


class FeatureVectorSpec(ModelObj):
    def __init__(
        self,
        features=None,
        description=None,
        entity_source=None,
        entity_fields=None,
        timestamp_field=None,
        graph=None,
        label_feature=None,
        with_indexes=None,
        function=None,
        analysis=None,
        relations=None,
        join_graph=None,
    ):
        self._graph: RootFlowStep = None
        self._entity_fields: ObjectList = None
        self._entity_source: DataSource = None
        self._function: FunctionReference = None
        self._relations: dict[str, ObjectDict] = None
        self._join_graph: JoinGraph = None

        self.description = description
        self.features: list[str] = features or []
        self.entity_source = entity_source
        self.entity_fields = entity_fields or []
        self.graph = graph
        self.join_graph = join_graph
        self.relations: dict[str, dict[str, Union[Entity, str]]] = relations or {}
        self.timestamp_field = timestamp_field
        self.label_feature = label_feature
        self.with_indexes = with_indexes
        self.function = function
        self.analysis = analysis or {}

    @property
    def entity_source(self) -> DataSource:
        """data source used as entity source (events/keys need to be enriched)"""
        return self._entity_source

    @entity_source.setter
    def entity_source(self, source: DataSource):
        self._entity_source = self._verify_dict(source, "entity_source", DataSource)

    @property
    def entity_fields(self) -> list[Feature]:
        """the schema/metadata for the entity source fields"""
        return self._entity_fields

    @entity_fields.setter
    def entity_fields(self, entity_fields: list[Feature]):
        self._entity_fields = ObjectList.from_list(Feature, entity_fields)

    @property
    def graph(self) -> RootFlowStep:
        """feature vector transformation graph/DAG"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowStep)
        self._graph.engine = "async"

    @property
    def function(self) -> FunctionReference:
        """reference to template graph processing function"""
        return self._function

    @function.setter
    def function(self, function):
        self._function = self._verify_dict(function, "function", FunctionReference)

    @property
    def relations(self) -> dict[str, ObjectDict]:
        """feature set relations dict"""
        return self._relations

    @relations.setter
    def relations(self, relations: dict[str, dict[str, Union[Entity, str]]]):
        temp_relations = {}
        for fs_name, relation in relations.items():
            for col, ent in relation.items():
                if isinstance(ent, str):
                    relation[col] = Entity(ent)
            temp_relations[fs_name] = ObjectDict.from_dict(
                {"entity": Entity}, relation, "entity"
            )
        self._relations = ObjectDict.from_dict(
            {"object_dict": ObjectDict}, temp_relations, "object_dict"
        )

    @property
    def join_graph(self):
        return self._join_graph

    @join_graph.setter
    def join_graph(self, join_graph):
        if join_graph is not None:
            self._join_graph = self._verify_dict(join_graph, "join_graph", JoinGraph)
        else:
            self._join_graph = None


class FeatureVectorStatus(ModelObj):
    def __init__(
        self,
        state=None,
        targets=None,
        features=None,
        label_column=None,
        stats=None,
        preview=None,
        run_uri=None,
        index_keys=None,
        timestamp_key=None,
    ):
        self._targets: ObjectList = None
        self._features: ObjectList = None

        self.state = state or "created"
        self.label_column = label_column
        self.targets = targets
        self.stats = stats or {}
        self.index_keys = index_keys
        self.preview = preview or []
        self.features: list[Feature] = features or []
        self.run_uri = run_uri
        self.timestamp_key = timestamp_key

    @property
    def targets(self) -> list[DataTarget]:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: list[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    @property
    def features(self) -> list[Feature]:
        """list of features (result of joining features from the source feature sets)"""
        return self._features

    @features.setter
    def features(self, features: list[Feature]):
        self._features = ObjectList.from_list(Feature, features)


class FixedWindowType(Enum):
    CurrentOpenWindow = 1
    LastClosedWindow = 2

    def to_qbk_fixed_window_type(self):
        try:
            from storey import FixedWindowType as QueryByKeyFixedWindowType
        except ImportError as exc:
            raise ImportError("storey not installed, use pip install storey") from exc
        if self == FixedWindowType.LastClosedWindow:
            return QueryByKeyFixedWindowType.LastClosedWindow
        elif self == FixedWindowType.CurrentOpenWindow:
            return QueryByKeyFixedWindowType.CurrentOpenWindow
        else:
            raise NotImplementedError(
                f"Provided fixed window type is not supported. fixed_window_type={self}"
            )


class FeatureVector(ModelObj):
    """Feature vector, specify selected features, their metadata and material views"""

    kind = mlrun.common.schemas.ObjectKind.feature_vector.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name=None,
        features=None,
        label_feature=None,
        description=None,
        with_indexes=None,
        join_graph: JoinGraph = None,
        relations: typing.Optional[dict[str, dict[str, Union[Entity, str]]]] = None,
    ):
        """Feature vector, specify selected features, their metadata and material views

        example::

            import mlrun.feature_store as fstore

            features = ["quotes.bid", "quotes.asks_sum_5h as asks_5h", "stocks.*"]
            vector = fstore.FeatureVector("my-vec", features)

            # get the vector as a dataframe
            df = vector.get_offline_features().to_dataframe()

            # return an online/real-time feature service
            svc = vector.get_online_feature_service(impute_policy={"*": "$mean"})
            resp = svc.get([{"stock": "GOOG"}])

        :param name:           List of names of targets to delete (default: delete all ingested targets)
        :param features:       list of feature to collect to this vector.
                                Format [<project>/]<feature_set>.<feature_name or `*`> [as <alias>]
        :param label_feature:  feature name to be used as label data
        :param description:    text description of the vector
        :param with_indexes:   whether to keep the entity and timestamp columns in the response
        :param join_graph:     An optional JoinGraph object representing the graph of data joins
                               between feature sets for this feature vector, specified the order and the join types.
        :param relations:      {<feature_set name>: {<column_name>: <other entity object/name>, ...}...}
                               An optional dictionary specifying the relations between feature sets in the
                               feature vector. The keys of the dictionary are feature set names, and the values
                               are dictionaries where the keys represent column names(of the feature set),
                               and the values represent the target entities to join with.
                               The relations provided here will take precedence over the relations that were specified
                               on the feature sets themselves. In case a specific feature set is not mentioned as a key
                               here, the function will fall back to using the default relations defined in the
                               feature set.

        """
        self._spec: FeatureVectorSpec = None
        self._metadata = None
        self._status = None

        self.spec = FeatureVectorSpec(
            description=description,
            features=features,
            label_feature=label_feature,
            with_indexes=with_indexes,
            relations=relations,
            join_graph=join_graph,
        )
        self.metadata = VersionedObjMetadata(name=name)
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
    def metadata(self) -> VersionedObjMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", VersionedObjMetadata)

    @property
    def status(self) -> FeatureVectorStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureVectorStatus)

    @property
    def uri(self):
        """fully qualified feature vector uri"""
        uri = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
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
            feature_aliases = self.get_feature_aliases()
            for old_name, new_name in feature_aliases.items():
                if old_name in self.status.stats:
                    self.status.stats[new_name] = self.status.stats[old_name]
                    del self.status.stats[old_name]
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def get_feature_aliases(self):
        feature_aliases = {}
        for feature in self.spec.features:
            column_names = feature.split(" as ")
            # split 'feature_set.old_name as new_name'
            if len(column_names) == 2:
                old_name_with_feature_set, new_name = column_names
                # split 'feature_set.old_name'
                feature_set, old_name = column_names[0].split(".")
                if new_name != old_name:
                    feature_aliases[old_name] = new_name
        return feature_aliases

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

    def reload(self, update_spec=True):
        """reload/sync the feature set status and spec from the DB"""
        from_db = mlrun.get_run_db().get_feature_vector(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        self.status = from_db.status
        if update_spec:
            self.spec = from_db.spec

    def parse_features(self, offline=True, update_stats=False):
        """parse and validate feature list (from vector) and add metadata from feature sets

        :returns
            feature_set_objects: cache of used feature set objects
            feature_set_fields:  list of field (name, alias) per featureset
        """
        processed_features = {}  # dict of name to (featureset, feature object)
        feature_set_objects = self.feature_set_objects or {}
        index_keys = []
        feature_set_fields = collections.defaultdict(list)
        features = copy(self.spec.features)
        label_column_name = None
        label_column_fset = None
        if offline and self.spec.label_feature:
            features.append(self.spec.label_feature)
            feature_set, name, _ = parse_feature_string(self.spec.label_feature)
            self.status.label_column = name
            label_column_name = name
            label_column_fset = feature_set

        def add_feature(name, alias, feature_set_object, feature_set_full_name):
            if alias in processed_features.keys():
                logging.log(
                    logging.WARN,
                    f"feature name/alias {alias} already specified,"
                    " you need to use another alias (feature-set.name [as alias])"
                    f" by default it changed to be {alias}_{feature_set_full_name}",
                )
                alias = f"{alias}_{feature_set_full_name}"

            feature = feature_set_object[name]
            processed_features[alias or name] = (feature_set_object, feature)
            feature_set_fields[feature_set_full_name].append((name, alias))

        for feature in features:
            project_name, feature = parse_project_name_from_feature_string(feature)
            feature_set, feature_name, alias = parse_feature_string(feature)
            if feature_set not in feature_set_objects.keys():
                feature_set_objects[feature_set] = get_feature_set_by_uri(
                    feature_set,
                    project_name if project_name is not None else self.metadata.project,
                )
            feature_set_object = feature_set_objects[feature_set]

            feature_fields = feature_set_object.spec.features.keys()
            if feature_name == "*":
                for field in feature_fields:
                    if field != feature_set_object.spec.timestamp_key and not (
                        feature_set == label_column_fset and field == label_column_name
                    ):
                        if alias:
                            add_feature(
                                field,
                                alias + "_" + field,
                                feature_set_object,
                                feature_set,
                            )
                        else:
                            add_feature(field, field, feature_set_object, feature_set)
            else:
                if feature_name not in feature_fields:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"feature {feature} not found in feature set {feature_set}"
                    )
                add_feature(feature_name, alias, feature_set_object, feature_set)

        for feature_set_name, fields in feature_set_fields.items():
            feature_set = feature_set_objects[feature_set_name]
            for key in feature_set.spec.entities.keys():
                if key not in index_keys:
                    index_keys.append(key)
            for name, alias in fields:
                if name in feature_set.status.stats and update_stats:
                    self.status.stats[name] = feature_set.status.stats[name]
                if name in feature_set.spec.features.keys():
                    feature = feature_set.spec.features[name].copy()
                    feature.origin = f"{feature_set.fullname}.{name}"
                    feature.name = alias or name
                    self.status.features[alias or name] = feature

        self.status.index_keys = index_keys
        return feature_set_objects, feature_set_fields

    def get_feature_set_relations(self, feature_set: Union[str, FeatureSet]):
        if isinstance(feature_set, str):
            feature_set = get_feature_set_by_uri(
                feature_set,
                self.metadata.project,
            )
        name = feature_set.metadata.name
        feature_set_relations = feature_set.spec.relations or {}
        if self.spec.relations and name in self.spec.relations:
            feature_set_relations = self.spec.relations[name]
        return feature_set_relations

    def get_offline_features(
        self,
        entity_rows=None,
        entity_timestamp_column: typing.Optional[str] = None,
        target: DataTargetBase = None,
        run_config: RunConfig = None,
        drop_columns: typing.Optional[list[str]] = None,
        start_time: typing.Optional[Union[str, datetime]] = None,
        end_time: typing.Optional[Union[str, datetime]] = None,
        with_indexes: bool = False,
        update_stats: bool = True,
        engine: typing.Optional[str] = None,
        engine_args: typing.Optional[dict] = None,
        query: typing.Optional[str] = None,
        order_by: typing.Optional[Union[str, list[str]]] = None,
        spark_service: typing.Optional[str] = None,
        timestamp_for_filtering: typing.Optional[Union[str, dict[str, str]]] = None,
        additional_filters: typing.Optional[list] = None,
    ):
        """retrieve offline feature vector results

        specify a feature vector object/uri and retrieve the desired features, their metadata
        and statistics. returns :py:class:`~mlrun.feature_store.OfflineVectorResponse`,
        results can be returned as a dataframe or written to a target

        The start_time and end_time attributes allow filtering the data to a given time range, they accept
        string values or pandas `Timestamp` objects, string values can also be relative, for example:
        "now", "now - 1d2h", "now+5m", where a valid pandas Timedelta string follows the verb "now",
        for time alignment you can use the verb "floor" e.g. "now -1d floor 1H" will align the time to the last hour
        (the floor string is passed to pandas.Timestamp.floor(), can use D, H, T, S for day, hour, min, sec alignment).
        Another option to filter the data is by the `query` argument - can be seen in the example.
        example::

            features = [
                "stock-quotes.bid",
                "stock-quotes.asks_sum_5h",
                "stock-quotes.ask as mycol",
                "stocks.*",
            ]
            vector = FeatureVector(features=features)
            vector.get_offline_features(entity_rows=trades, entity_timestamp_column="time", query="ticker in ['GOOG']
              and bid>100")
            print(resp.to_dataframe())
            print(vector.get_stats_table())
            resp.to_parquet("./out.parquet")

        :param entity_rows:             dataframe with entity rows to join with
        :param target:                  where to write the results to
        :param drop_columns:            list of columns to drop from the final result
        :param entity_timestamp_column: timestamp column name in the entity rows dataframe. can be specified
                                        only if param entity_rows was specified.
        :param run_config:              function and/or run configuration
                                        see :py:class:`~mlrun.feature_store.RunConfig`
        :param start_time:              datetime, low limit of time needed to be filtered. Optional.
        :param end_time:                datetime, high limit of time needed to be filtered. Optional.
        :param with_indexes:            Return vector with/without the entities and the timestamp_key of the feature
                                        sets and with/without entity_timestamp_column and timestamp_for_filtering
                                        columns. This property can be specified also in the feature vector spec
                                        (feature_vector.spec.with_indexes)
                                        (default False)
        :param update_stats:            When set to True (default), updates feature statistics from the requested
                                        feature sets on the vector, which requires 'update' permissions. When set to
                                        False, uses read-only operations that only require 'read' permissions.
        :param engine:                  processing engine kind ("local", "dask", or "spark")
        :param engine_args:             kwargs for the processing engine
        :param query:                   The query string used to filter rows on the output
        :param spark_service:           Name of the spark service to be used (when using a remote-spark runtime)
        :param order_by:                Name or list of names to order by. The name or the names in the list can be the
                                        feature name or the alias of the feature you pass in the feature list.
        :param timestamp_for_filtering: name of the column to filter by, can be str for all the feature sets or a
                                        dictionary ({<feature set name>: <timestamp column name>, ...})
                                        that indicates the timestamp column name for each feature set. Optional.
                                        By default, the filter executes on the timestamp_key of each feature set.
                                        Note: the time filtering is performed on each feature set before the
                                        merge process using start_time and end_time params.
        :param additional_filters: List of additional_filter conditions as tuples.
                            Each tuple should be in the format (column_name, operator, value).
                            Supported operators: "=", ">=", "<=", ">", "<".
                            Example: [("Product", "=", "Computer")]
                            For all supported filters, please see:
                            https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html

        """
        if entity_rows is None and entity_timestamp_column is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "entity_timestamp_column param "
                "can not be specified without entity_rows param"
            )

        if isinstance(target, BaseStoreTarget) and not target.support_pandas:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"get_offline_features does not support targets that do not support pandas engine."
                f" Target kind: {target.kind}"
            )

        if update_stats:
            feature_vector = _features_to_vector_and_check_permissions(
                self, update_stats
            )
        else:
            feature_vector = self
            verify_feature_vector_permissions(
                feature_vector, mlrun.common.schemas.AuthorizationAction.read
            )

        entity_timestamp_column = (
            entity_timestamp_column or feature_vector.spec.timestamp_field
        )

        merger_engine = get_merger(engine)

        if run_config and not run_config.local:
            return run_merge_job(
                feature_vector,
                target,
                merger_engine,
                engine,
                engine_args,
                spark_service,
                entity_rows,
                entity_timestamp_column=entity_timestamp_column,
                run_config=run_config,
                drop_columns=drop_columns,
                with_indexes=with_indexes,
                query=query,
                order_by=order_by,
                start_time=start_time,
                end_time=end_time,
                timestamp_for_filtering=timestamp_for_filtering,
                additional_filters=additional_filters,
            )

        merger = merger_engine(feature_vector, **(engine_args or {}))
        return merger.start(
            entity_rows,
            entity_timestamp_column,
            target=target,
            drop_columns=drop_columns,
            start_time=start_time,
            end_time=end_time,
            timestamp_for_filtering=timestamp_for_filtering,
            with_indexes=with_indexes,
            update_stats=update_stats,
            query=query,
            order_by=order_by,
            additional_filters=additional_filters,
        )

    def get_online_feature_service(
        self,
        run_config: RunConfig = None,
        fixed_window_type: FixedWindowType = FixedWindowType.LastClosedWindow,
        impute_policy: typing.Optional[dict] = None,
        update_stats: bool = False,
        entity_keys: typing.Optional[list[str]] = None,
    ) -> OnlineVectorService:
        """initialize and return online feature vector service api,
        returns :py:class:`~mlrun.feature_store.OnlineVectorService`

        :**usage**:
            There are two ways to use the function:

            1. As context manager

                Example::

                    with vector_uri.get_online_feature_service() as svc:
                        resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
                        print(resp)
                        resp = svc.get([{"ticker": "AAPL"}], as_list=True)
                        print(resp)

                Example with imputing::

                    with vector_uri.get_online_feature_service(entity_keys=['id'],
                                                    impute_policy={"*": "$mean", "amount": 0)) as svc:
                        resp = svc.get([{"id": "C123487"}])

            2. as simple function, note that in that option you need to close the session.

                Example::

                    svc = vector_uri.get_online_feature_service(entity_keys=["ticker"])
                    try:
                        resp = svc.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}])
                        print(resp)
                        resp = svc.get([{"ticker": "AAPL"}], as_list=True)
                        print(resp)

                    finally:
                        svc.close()

                Example with imputing::

                    svc = vector_uri.get_online_feature_service(entity_keys=['id'],
                                                    impute_policy={"*": "$mean", "amount": 0))
                    try:
                        resp = svc.get([{"id": "C123487"}])
                    except Exception as e:
                        handling exception...
                    finally:
                        svc.close()

        :param run_config:          function and/or run configuration for remote jobs/services
        :param impute_policy:       a dict with `impute_policy` per feature, the dict key is the feature name and the
                                    dict value indicate which value will be used in case the feature is NaN/empty, the
                                    replaced value can be fixed number for constants or $mean, $max, $min, $std, $count
                                    for statistical values.
                                    "*" is used to specify the default for all features, example: `{"*": "$mean"}`
        :param fixed_window_type:   determines how to query the fixed window values which were previously inserted by
                                    ingest
        :param update_stats:        update features statistics from the requested feature sets on the vector.
                                    Default: False.
        :param entity_keys:         Entity list of the first feature_set in the vector.
                                    The indexes that are used to query the online service.
        :return:                    Initialize the `OnlineVectorService`.
                                    Will be used in subclasses where `support_online=True`.
        """
        feature_vector = _features_to_vector_and_check_permissions(self, True)

        engine_args = {"impute_policy": impute_policy}
        merger_engine = get_merger("storey")
        # todo: support remote service (using remote nuclio/mlrun function if run_config)

        merger = merger_engine(feature_vector, **engine_args)

        return merger.init_online_vector_service(
            entity_keys, fixed_window_type, update_stats=True
        )
