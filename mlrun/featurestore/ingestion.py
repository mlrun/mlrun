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

from storey import DataframeSource

from mlrun.featurestore.sources import get_source_driver
from mlrun.featurestore.targets import init_featureset_targets, add_target_states
from mlrun.datastore.data_resources import ResourceCache
from mlrun.serving.server import create_graph_server


def init_featureset_graph(
    df, featureset, namespace, with_targets=False, return_df=True,
):
    """create storey ingestion graph/DAG from feature set object"""

    cache = ResourceCache()
    targets = []
    graph = featureset.spec.graph.copy()
    _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)

    # init targets (and table)
    if with_targets:
        table = init_featureset_targets(featureset)
        if table:
            cache.cache_table(featureset.uri(), table, True)
        targets = featureset.spec.targets

    cache.cache_resource(featureset.uri(), featureset, True)
    add_target_states(
        graph, featureset, targets, to_df=return_df, final_state=default_final_state
    )

    # init source
    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    key_column = entity_columns[0]
    source = DataframeSource(df, key_column, featureset.spec.timestamp_key)
    graph.set_flow_source(source)

    server = create_graph_server(graph=graph, parameters={})
    server.init(None, namespace, cache)
    return graph._controller


def featureset_initializer(server):
    """graph server hook to initialize feature set ingestion graph/DAG"""

    context = server.context
    cache = server.resource_cache
    graph = server.graph

    featureset_uri = context.get_param("featureset")
    featureset = context.get_data_resource(featureset_uri)

    table = init_featureset_targets(featureset)
    if table:
        cache.cache_table(featureset.uri(), table, True)
    cache.cache_resource(featureset.uri(), featureset, True)

    targets = featureset.spec.targets
    _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)
    add_target_states(graph, featureset, targets, final_state=default_final_state)

    source = get_source_driver(featureset.spec.source)
    graph.set_flow_source(source)
