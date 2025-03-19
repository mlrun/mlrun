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
import mlrun
from mlrun.datastore.store_resources import ResourceCache
from mlrun.datastore.targets import get_online_target
from mlrun.serving.server import create_graph_server

from ..feature_vector_utils import OnlineVectorService
from .base import BaseMerger


class StoreyFeatureMerger(BaseMerger):
    engine = "storey"
    support_online = True

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        self.impute_policy = engine_args.get("impute_policy")

    def _generate_online_feature_vector_graph(
        self,
        entity_keys,
        feature_set_fields,
        feature_set_objects,
        fixed_window_type,
    ):
        graph = self.vector.spec.graph.copy()
        start_states, default_final_state, responders = graph.check_and_process_graph(
            allow_empty=True
        )
        next = graph

        join_graph = self._get_graph(
            feature_set_objects, feature_set_fields, entity_keys
        )

        all_columns = []
        save_column = []
        entity_keys = []
        del_columns = []
        end_aliases = {}
        for step in join_graph.steps:
            name = step.right_feature_set_name
            feature_set = feature_set_objects[name]
            columns = feature_set_fields[name]
            column_names = [name for name, alias in columns]
            aliases = {name: alias for name, alias in columns if alias}
            all_columns += [aliases.get(name, name) for name in column_names]
            saved_columns_for_relation = list(
                self.vector.get_feature_set_relations(feature_set).keys()
            )

            for col in saved_columns_for_relation:
                if col not in column_names:
                    column_names.append(col)
                    del_columns.append(col)
                else:
                    save_column.append(col)

            entity_list = step.right_keys or list(feature_set.spec.entities.keys())
            if not entity_keys:
                # if entity_keys not provided by the user we will set it to be the entity of the first feature set
                entity_keys = entity_list
            end_aliases.update(
                {
                    k: v
                    for k, v in zip(entity_list, step.left_keys)
                    if k != v and v in save_column
                }
            )
            mapping = {k: v for k, v in zip(step.left_keys, entity_list) if k != v}
            if mapping:
                next = next.to(
                    "storey.Rename",
                    f"rename-{name}",
                    mapping=mapping,
                )

            next = next.to(
                "storey.QueryByKey",
                f"query-{name}",
                features=column_names,
                table=feature_set.uri,
                key_field=entity_list,
                aliases=aliases,
                fixed_window_type=fixed_window_type.to_qbk_fixed_window_type(),
            )
        if end_aliases:
            # run if the user want to save a column that related to another entity
            next = next.to(
                "storey.Rename",
                "rename-entity-to-features",
                mapping=end_aliases,
            )
        if del_columns:
            next = next.to(
                "storey.flow.DropColumns",
                "drop-unnecessary-columns",
                columns=del_columns,
            )
        for name in start_states:
            next.set_next(name)

        if not start_states:  # graph was empty
            next.respond()
        elif not responders and default_final_state:  # graph has clear state sequence
            graph[default_final_state].respond()
        elif not responders:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "the graph doesnt have an explicit final step to respond on"
            )
        return graph, all_columns, entity_keys

    def init_online_vector_service(
        self, entity_keys, fixed_window_type, update_stats=False
    ):
        try:
            from storey import SyncEmitSource
        except ImportError as exc:
            raise ImportError(f"storey not installed, use pip install storey, {exc}")

        feature_set_objects, feature_set_fields = self.vector.parse_features(
            offline=False, update_stats=update_stats
        )
        if not feature_set_fields:
            raise mlrun.errors.MLRunRuntimeError(
                f"No features found for feature vector '{self.vector.metadata.name}'"
            )
        (
            graph,
            requested_columns,
            entity_keys,
        ) = self._generate_online_feature_vector_graph(
            entity_keys,
            feature_set_fields,
            feature_set_objects,
            fixed_window_type,
        )
        graph.set_flow_source(SyncEmitSource())
        server = create_graph_server(graph=graph, parameters={})

        cache = ResourceCache()
        for featureset in feature_set_objects.values():
            driver = get_online_target(featureset)
            if not driver:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"resource {featureset.uri} does not have an online data target"
                )
            cache.cache_table(featureset.uri, driver.get_table_object())

        server.init_states(context=None, namespace=None, resource_cache=cache)
        server.init_object(None)

        service = OnlineVectorService(
            self.vector,
            graph,
            entity_keys,
            impute_policy=self.impute_policy,
            requested_columns=requested_columns,
        )
        service.initialize()

        return service
