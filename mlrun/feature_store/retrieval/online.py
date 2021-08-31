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


import mlrun
from mlrun.datastore.store_resources import ResourceCache
from mlrun.datastore.targets import get_online_target
from mlrun.serving.server import create_graph_server


def _build_feature_vector_graph(
    vector, feature_set_fields, feature_set_objects, fixed_window_type,
):
    graph = vector.spec.graph.copy()
    start_states, default_final_state, responders = graph.check_and_process_graph(
        allow_empty=True
    )
    next = graph

    for name, columns in feature_set_fields.items():
        featureset = feature_set_objects[name]
        column_names = [name for name, alias in columns]
        aliases = {name: alias for name, alias in columns if alias}

        entity_list = list(featureset.spec.entities.keys())
        next = next.to(
            "storey.QueryByKey",
            f"query-{name}",
            features=column_names,
            table=featureset.uri,
            key=entity_list,
            aliases=aliases,
            fixed_window_type=fixed_window_type.to_qbk_fixed_window_type(),
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
    return graph


def init_feature_vector_graph(vector, query_options):
    try:
        from storey import SyncEmitSource
    except ImportError as exc:
        raise ImportError(f"storey not installed, use pip install storey, {exc}")

    feature_set_objects, feature_set_fields = vector.parse_features(False)
    graph = _build_feature_vector_graph(
        vector, feature_set_fields, feature_set_objects, query_options
    )
    graph.set_flow_source(SyncEmitSource())
    server = create_graph_server(graph=graph, parameters={})

    cache = ResourceCache()
    index_columns = []
    for featureset in feature_set_objects.values():
        driver = get_online_target(featureset)
        if not driver:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"resource {featureset.uri} does not have an online data target"
            )
        cache.cache_table(featureset.uri, driver.get_table_object())
        for key in featureset.spec.entities.keys():
            if not vector.spec.with_indexes and key not in index_columns:
                index_columns.append(key)
    server.init_states(context=None, namespace=None, resource_cache=cache)
    server.init_object(None)
    return graph, index_columns
