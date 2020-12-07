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
import inspect
from copy import copy

from .model import (
    FeatureSetStatus,
    FeatureSetSpec,
    FeatureSetMetadata,
    FeatureAggregation,
    Feature,
)
from .infer import infer_schema_from_df, get_df_stats, get_df_preview
from .pipeline import init_featureset_graph
from .targets import add_target_states
from ..model import ModelObj
from ..serving.states import ServingTaskState


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
        self._init_graph()

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

        namespace = namespace or inspect.stack()[1][0].f_globals
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
            self._status.stats = get_df_stats(df, with_histogram)
        if with_preview:
            self._status.preview = get_df_preview(df)
        if label_column:
            self._spec.label_column = label_column
        return df

    def add_entity(self, entity, name=None):
        self._spec.entities.update(entity, name)

    def add_feature(self, feature, name=None):
        self._spec.features.update(feature, name)

    def add_flow_step(self, name, class_name, after=None, **class_args):
        graph = self._spec.graph
        before = "Aggregates" if "Aggregates" in graph.states else "ValidatorStep"
        return graph.add_state(
            name,
            ServingTaskState(class_name, class_args=class_args),
            after=after or "$prev",
            before=before,
        )

    def _init_graph(self):
        graph = self._spec.graph
        if "ValidatorStep" not in graph.states:
            graph.add_state(
                "ValidatorStep",
                after="$last",
                class_name="mlrun.featurestore.ValidatorStep",
                featureset=".",
            )
        graph._last_added = None

    def add_aggregation(
        self, name, column, operations, windows, period=None, state_name=None
    ):
        aggregation = FeatureAggregation(
            name, column, operations, windows, period
        ).to_dict()
        # self._spec.aggregations.update(aggregation)

        def upsert_feature(name):
            if name in self.spec.features:
                self.spec.features[name].aggregate = True
            else:
                self.spec.features[name] = Feature(name=column, aggregate=True)

        state_name = state_name or "Aggregates"
        graph = self.spec.graph
        if state_name in graph.states:
            state = graph.states[state_name]
            aggregations = state.class_args.get("aggregates", [])
            aggregations.append(aggregation)
            state.class_args["aggregates"] = aggregations
        else:
            last_state = graph._last_added
            # start_at = graph.start_at
            graph.add_state(
                state_name,
                after=last_state.name if last_state else None,
                before="ValidatorStep",
                class_name="storey.AggregateByKey",
                aggregates=[aggregation],
                table=".",
            )
            graph._last_added = last_state
            # graph.start_at = start_at

        for operation in operations:
            for window in windows:
                upsert_feature(f"{name}_{operation}_{window}")

    def __getitem__(self, name):
        return self._spec.features[name]

    def __setitem__(self, key, item):
        self._spec.features.update(item, key)

    def merge(self, other):
        pass

    def plot(self, filename=None, format=None, with_targets=False, **kw):
        graph = self.spec.graph
        if with_targets:
            graph = copy(graph)
            add_target_states(graph, self, self.spec.targets)
        return graph.plot(filename, format, **kw)
