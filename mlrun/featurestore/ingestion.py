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

from .model import DataTarget
from .model.base import DataTargetSpec
from .sources import get_source_step
from .targets import add_target_states
from mlrun.datastore.store_resources import ResourceCache
from mlrun.serving.server import create_graph_server


def init_featureset_graph(source, featureset, namespace, targets=None, return_df=True):
    """create storey ingestion graph/DAG from feature set object"""

    cache = ResourceCache()
    graph = featureset.spec.graph.copy()
    _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)

    # init targets (and table)
    targets = targets or []
    cache.cache_resource(featureset.uri(), featureset, True)
    table = add_target_states(
        graph, featureset, targets, to_df=return_df, final_state=default_final_state
    )
    if table:
        cache.cache_table(featureset.uri(), table, True)

    # init source
    entity_columns = list(featureset.spec.entities.keys())
    if not entity_columns:
        raise ValueError("entity column(s) are not defined in feature set")
    key_column = entity_columns[0]
    source = get_source_step(
        source, key_column, time_column=featureset.spec.timestamp_key
    )
    graph.set_flow_source(source)

    server = create_graph_server(graph=graph, parameters={})
    server.init(None, namespace, cache)
    return graph._controller


def featureset_initializer(server):
    """graph server hook to initialize feature set ingestion graph/DAG"""

    context = server.context
    cache = server.resource_cache

    featureset_uri = context.get_param("featureset")
    source = context.get_param("source")
    featureset = context.get_store_resource(featureset_uri)
    targets = context.get_param("targets", None)
    if targets:
        targets = [DataTarget.from_dict(target) for target in targets]
    else:
        targets = featureset.spec.targets

    graph = featureset.spec.graph.copy()
    server.graph = graph
    _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)
    table = add_target_states(
        graph, featureset, targets, final_state=default_final_state
    )
    cache.cache_resource(featureset_uri, featureset, True)
    if table:
        cache.cache_table(featureset_uri, table, True)

    entity_columns = list(featureset.spec.entities.keys())
    key_column = context.get_param("key_column", entity_columns[0])
    time_column = context.get_param("time_column", featureset.spec.timestamp_key)
    source = get_source_step(source, key_column=key_column, time_column=time_column)
    graph.set_flow_source(source)


def run_ingestion_function(featureset, source, targets: List[DataTargetSpec] = None):
    parameters = {"featureset": featureset.uri(), "source": source.to_dict()}
    if targets:
        parameters["targets"] = [target.to_dict() for target in targets]
    elif not featureset.spec.targets:
        featureset.set_targets()
    featureset.save()

    server = create_graph_server(parameters=parameters)
    server.graph_initializer = "mlrun.featurestore.ingestion.featureset_initializer"
    server.verbose = True
    server.init(None, None)
    server.graph.controller.await_termination()
