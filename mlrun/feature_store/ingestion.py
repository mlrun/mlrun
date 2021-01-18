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
from mlrun.runtimes import RuntimeKinds

from .model import DataTarget
from .sources import get_source_step
from .targets import add_target_states
from mlrun.datastore.store_resources import ResourceCache
from mlrun.serving.server import create_graph_server
from ..runtimes.function_reference import FunctionReference


def init_featureset_graph(source, featureset, namespace, targets=None, return_df=True):
    """create storey ingestion graph/DAG from feature set object"""

    cache = ResourceCache()
    graph = featureset.spec.graph.copy()

    # init targets (and table)
    targets = targets or []
    _add_data_states(
        graph, cache, featureset, targets=targets, source=source, return_df=return_df,
    )

    server = create_graph_server(graph=graph, parameters={})
    server.init(None, namespace, cache)
    return graph


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
    _add_data_states(
        graph, cache, featureset, targets=targets, source=source,
    )
    server.graph = graph


def _add_data_states(
    graph, cache, featureset, targets, source, return_df=False,
):
    _, default_final_state, _ = graph.check_and_process_graph(allow_empty=True)
    cache.cache_resource(featureset.uri, featureset, True)
    table = add_target_states(
        graph, featureset, targets, to_df=return_df, final_state=default_final_state
    )
    if table:
        cache.cache_table(featureset.uri, table, True)

    entity_columns = list(featureset.spec.entities.keys())
    key_column = entity_columns[0] if entity_columns else None
    if source is not None:
        source = get_source_step(
            source, key_column=key_column, time_column=featureset.spec.timestamp_key,
        )
    graph.set_flow_source(source)


def deploy_ingestion_function(
    name, source, featureset, parameters, function_ref=None, local=False, watch=True
):
    name = name or f"{featureset.metadata.name}_ingest"
    function_ref = function_ref or featureset.spec.function.copy()
    if not function_ref.to_dict():
        runtime_kind = RuntimeKinds.serving if source.online else RuntimeKinds.job
        function_ref = FunctionReference(name=name, kind=runtime_kind)
    if not function_ref.kind:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"function reference is missing kind {function_ref}"
        )

    if function_ref.kind == RuntimeKinds.serving:
        function_ref.code = function_ref.code or ""
    elif function_ref.kind == RuntimeKinds.spark:
        function_ref.code = function_ref.code or _default_spark_handler
        # todo: use spark specific image
    else:
        function_ref.code = function_ref.code or _default_job_handler

    function_ref.image = function_ref.image or "mlrun/mlrun"
    function = function_ref.to_function()
    function.metadata.project = featureset.metadata.project

    if function.kind == RuntimeKinds.serving:
        # add triggers
        function.spec.parameters = parameters
        function.spec.graph_initializer = (
            "mlrun.feature_store.ingestion.featureset_initializer"
        )
        function.verbose = True
        if local:
            return function.to_mock_server()
        else:
            function.deploy()
    else:
        return function.run(
            params=parameters, schedule=source.schedule, local=local, watch=watch
        )


_default_job_handler = """
import mlrun
def handler(context):
    verbose = context.get_param('verbose', True)
    server = mlrun.serving.create_graph_server(parameters=context.parameters, verbose=verbose)
    server.graph_initializer = "mlrun.feature_store.ingestion.featureset_initializer"
    server.init(None, globals())
    server.wait_for_completion()
"""


_default_spark_handler = """
import mlrun
def handler(context):
    pass
    # todo: call to spark ingestion handler
"""
