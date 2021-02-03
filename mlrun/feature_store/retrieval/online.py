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
from mlrun.serving.server import create_graph_server
from ..targets import get_online_target


def _build_feature_vector_graph(
    vector, feature_set_fields, feature_set_objects,
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
        key_column = entity_list[0]
        next = next.to(
            "storey.QueryByKey",
            f"query-{name}",
            features=column_names,
            table=featureset.uri,
            key=key_column,
            aliases=aliases,
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


def init_feature_vector_graph(vector):
    try:
        from storey import Source
    except ImportError as e:
        raise ImportError(f"storey not installed, use pip install storey, {e}")

    feature_set_objects, feature_set_fields = vector.parse_features()
    graph = _build_feature_vector_graph(vector, feature_set_fields, feature_set_objects)
    graph.set_flow_source(Source())
    server = create_graph_server(graph=graph, parameters={})

    cache = ResourceCache()
    for featureset in feature_set_objects.values():
        driver = get_online_target(featureset)
        cache.cache_table(featureset.uri, driver.get_table_object())
    server.init(None, None, cache)
    return graph
