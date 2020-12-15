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
from copy import copy
import pandas as pd
from mlrun.utils import get_caller_globals

from .model import (
    FeatureSetStatus,
    FeatureSetSpec,
    FeatureSetMetadata,
    FeatureAggregation,
    Feature,
    store_config,
    DataTargetSpec,
)
from .infer import infer_schema_from_df, get_df_stats, get_df_preview
from .pipeline import init_featureset_graph
from .targets import init_target
from ..model import ModelObj
from ..serving.states import BaseState

aggregates_step = "Aggregates"


class FeatureSet(ModelObj):
    """Feature Set"""

    kind = "FeatureSet"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, description=None, entities=None, timestamp_key=None):
        self._spec: FeatureSetSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None

        self.spec = FeatureSetSpec(
            description=description, entities=entities, timestamp_key=timestamp_key
        )
        self.metadata = FeatureSetMetadata(name=name)
        self.status = None
        self._last_state = ""

    @property
    def spec(self) -> FeatureSetSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureSetSpec)

    @property
    def metadata(self) -> FeatureSetMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", FeatureSetMetadata)

    @property
    def status(self) -> FeatureSetStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureSetStatus)

    def uri(self):
        uri = f'{self._metadata.project or ""}/{self._metadata.name}'
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def infer_from_df(
        self,
        df,
        with_stats=False,
        entity_columns=None,
        timestamp_key=None,
        label_column=None,
        with_index=True,
        with_histogram=False,
        with_preview=False,
        namespace=None,
        client=None,
    ):
        """Infer features schema and stats from a local DataFrame"""
        if timestamp_key is not None:
            self._spec.timestamp_key = timestamp_key

        namespace = namespace or get_caller_globals()
        if self._spec.require_processing():
            # find/update entities schema
            infer_schema_from_df(
                df, self._spec, entity_columns, with_index, with_features=False
            )
            controller = init_featureset_graph(
                df, self, namespace, client, with_targets=False, return_df=True
            )
            df = controller.await_termination()
            # df = ingest_from_df(context, self, df, namespace=namespace).await_termination()

        infer_schema_from_df(df, self._spec, entity_columns, with_index)
        if with_stats:
            self._status.stats = get_df_stats(df, with_histogram=with_histogram)
        if with_preview:
            self._status.preview = get_df_preview(df)
        if label_column:
            self._spec.label_column = label_column
        return df

    def set_targets(self, targets=None):
        if targets is not None and not isinstance(targets, list):
            raise ValueError(
                "targets can only be None or a list of kinds/DataTargetSpec"
            )
        targets = targets or copy(store_config.default_targets)
        for target in targets:
            if not isinstance(target, DataTargetSpec):
                target = DataTargetSpec(target, name=str(target))
            init_target(self, target)
            self.spec.targets.update(target)

    def add_entity(self, entity, name=None):
        self._spec.entities.update(entity, name)

    def add_feature(self, feature, name=None):
        self._spec.features.update(feature, name)

    @property
    def graph(self):
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
            # last_state = graph._last_added
            # start_at = graph.start_at
            graph.add_step(
                name=state_name,
                after=after or "$prev",
                before=before,
                class_name="storey.AggregateByKey",
                aggregates=[aggregation],
                table=".",
            )
            # graph._last_added = last_state
            # graph.start_at = start_at

        for operation in operations:
            for window in windows:
                upsert_feature(f"{name}_{operation}_{window}")

    def get_stats_table(self):
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def __getitem__(self, name):
        return self._spec.features[name]

    def __setitem__(self, key, item):
        self._spec.features.update(item, key)

    def merge(self, other):
        pass

    def plot(self, filename=None, format=None, with_targets=False, **kw):
        graph = self.spec.graph
        targets = None
        if with_targets:
            targets = [
                BaseState(target.kind, shape="cylinder") for target in self.spec.targets
            ]
        return graph.plot(filename, format, targets=targets, **kw)
