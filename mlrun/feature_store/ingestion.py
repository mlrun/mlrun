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
from ..runtimes import RuntimeKinds

from mlrun.datastore.sources import get_source_step, get_source_from_dict
from mlrun.datastore.targets import add_target_states, get_target_driver
from ..datastore.store_resources import ResourceCache
from ..serving.server import create_graph_server
from ..data_types import InferOptions
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
    featureset, source, targets, infer_options = context_to_ingestion_params(context)
    graph = featureset.spec.graph.copy()
    _add_data_states(
        graph, cache, featureset, targets=targets, source=source,
    )
    featureset.save()
    server.graph = graph


def context_to_ingestion_params(context):
    """extract the ingestion task params from job/serving context"""

    featureset_uri = context.get_param("featureset")
    featureset = context.get_store_resource(featureset_uri)
    infer_options = context.get_param("infer_options", InferOptions.Null)

    source = context.get_param("source")
    if source:
        source = get_source_from_dict(source)
    elif featureset.spec.source.to_dict():
        source = get_source_from_dict(featureset.spec.source.to_dict())

    targets = context.get_param("targets", None)
    if not targets:
        targets = featureset.spec.targets
    targets = [get_target_driver(target, featureset) for target in targets]
    return featureset, source, targets, infer_options


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
    key_field = entity_columns[0] if entity_columns else None
    if source is not None:
        source = get_source_step(
            source, key_field=key_field, time_field=featureset.spec.timestamp_key,
        )
    graph.set_flow_source(source)


def default_ingestion_job_function(name, featureset, engine=None, function=None):
    if not function:
        function_ref = featureset.spec.function.copy()
        if not function_ref.to_dict():
            function_ref = FunctionReference(name=name, kind=RuntimeKinds.job)
        function_ref.kind = function_ref.kind or RuntimeKinds.job

        if not function_ref.url:
            code = function_ref.code or ""
            # todo: use engine until we have spark service runtime
            engine = engine or featureset.spec.engine
            if engine and engine == "spark":
                function_ref.code = code + _default_spark_handler
            else:
                function_ref.code = code + _default_job_handler
        function = function_ref.to_function()
        function.spec.default_handler = "handler"

    if not function.spec.image:
        function.spec.image = (
            _default_spark_image()
            if engine == "spark"
            else mlrun.mlconf.feature_store.default_job_image
        )
        if not function.spec.image:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "function image must be specified"
            )
    return function


def _default_spark_image():
    image = mlrun.mlconf.spark_app_image
    if mlrun.mlconf.spark_app_image_tag:
        image += ":" + mlrun.mlconf.spark_app_image_tag
    return image


_default_job_handler = """
from mlrun.feature_store.api import ingest
def handler(context):
    ingest(mlrun_context=context)
"""


_default_spark_handler = """
from mlrun.feature_store.api import ingest_with_spark
def handler(context):
    ingest_with_spark(mlrun_context=context)
"""
