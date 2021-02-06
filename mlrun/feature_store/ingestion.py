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

from .sources import get_source_step, get_source_from_dict
from .targets import add_target_states, get_target_driver
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
    key_column = entity_columns[0] if entity_columns else None
    if source is not None:
        source = get_source_step(
            source, key_column=key_column, time_column=featureset.spec.timestamp_key,
        )
    graph.set_flow_source(source)


def deploy_ingestion_function(
    name, featureset, source, parameters, function=None, local=False, watch=True,
):
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
            name=name,
            params=parameters,
            schedule=source.schedule,
            local=local,
            watch=watch,
        )


def default_ingestion_function(name, featureset, online, engine=None):
    name = name or f"{featureset.metadata.name}_ingest"
    function_ref = featureset.spec.function.copy()
    if not function_ref.to_dict():
        runtime_kind = RuntimeKinds.serving if online else RuntimeKinds.job
        function_ref = FunctionReference(name=name, kind=runtime_kind)
    if not function_ref.kind:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"function reference is missing kind {function_ref}"
        )

    if not function_ref.url:
        code = function_ref.code or ""
        if function_ref.kind == RuntimeKinds.serving:
            function_ref.code = code
        else:
            engine = engine or featureset.spec.engine
            if engine and engine == "spark":
                function_ref.code = code + _default_spark_handler
            else:
                function_ref.code = code + _default_job_handler

    if not function_ref.image:
        function_ref.image = (
            mlrun.mlconf.feature_store.default_spark_image
            if engine == "spark"
            else mlrun.mlconf.feature_store.default_job_image
        )
    function = function_ref.to_function()
    return name, function


_default_job_handler = """
from mlrun.feature_store.ingestion import context_to_ingestion_params
from mlrun.feature_store.api import ingest
def handler(context):
    featureset, source, targets, infer_options = context_to_ingestion_params(context)
    context.logger.info(f"starting ingestion task to {featureset.uri}")
    ingest(featureset, source, targets, globals(), return_df=False, infer_options=infer_options)
    context.logger.info("ingestion task completed, targets:")
    context.logger.info(f"{featureset.status.targets.to_dict()}")
    context.log_result('featureset', featureset.uri)
"""


_default_spark_handler = """
from mlrun.feature_store.ingestion import context_to_ingestion_params
from mlrun.feature_store.api import spark_ingestion, spark_transform_handler
def spark_job_handler(context):
    featureset, source, targets, infer_options = context_to_ingestion_params(context)
    context.logger.info(f"starting ingestion task to {featureset.uri}")
    if not source:
        raise ValueError("data source was not specified")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName(f"{context.name}-{context.uid}").getOrCreate()
    handler = globals().get(spark_transform_handler, None)
    spark_ingestion(spark, featureset, source, targets, infer_options,
                    mlrun_context=context, transformer=handler)
    context.logger.info("ingestion task completed, targets:")
    context.logger.info(f"{featureset.status.targets.to_dict()}")
    context.log_result('featureset', featureset.uri)
    spark.stop()
"""
