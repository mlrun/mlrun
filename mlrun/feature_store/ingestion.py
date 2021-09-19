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

import uuid

import mlrun
from mlrun.datastore.sources import get_source_from_dict, get_source_step
from mlrun.datastore.targets import (
    add_target_steps,
    get_target_driver,
    validate_target_list,
    validate_target_placement,
)

from ..data_types import InferOptions
from ..datastore.store_resources import ResourceCache
from ..runtimes import RuntimeKinds
from ..runtimes.function_reference import FunctionReference
from ..serving.server import MockEvent, create_graph_server
from ..utils import logger


def init_featureset_graph(
    source, featureset, namespace, targets=None, return_df=True, verbose=False
):
    """create storey ingestion graph/DAG from feature set object"""

    cache = ResourceCache()
    graph = featureset.spec.graph.copy()

    # init targets (and table)
    targets = targets or []
    server = create_graph_server(graph=graph, parameters={}, verbose=verbose)
    server.init_states(context=None, namespace=namespace, resource_cache=cache)

    if graph.engine != "sync":
        _add_data_steps(
            graph,
            cache,
            featureset,
            targets=targets,
            source=source,
            return_df=return_df,
            context=server.context,
        )

    server.init_object(namespace)

    if graph.engine != "sync":
        return graph.wait_for_completion()

    if hasattr(source, "to_dataframe"):
        source = source.to_dataframe()
    elif not hasattr(source, "to_csv"):
        raise mlrun.errors.MLRunInvalidArgumentError("illegal source")

    event = MockEvent(body=source)
    data = server.run(event, get_body=True)
    for target in targets:
        target = get_target_driver(target, featureset)
        size = target.write_dataframe(data)
        target_status = target.update_resource_status("ready", size=size)
        if verbose:
            logger.info(f"wrote target: {target_status}")

    return data


def featureset_initializer(server):
    """graph server hook to initialize feature set ingestion graph/DAG"""

    context = server.context
    cache = server.resource_cache
    featureset, source, targets, _ = context_to_ingestion_params(context)
    graph = featureset.spec.graph.copy()
    _add_data_steps(
        graph, cache, featureset, targets=targets, source=source,
    )
    featureset.save()
    server.graph = graph


def run_spark_graph(df, featureset, namespace, spark):
    """run spark (sync) pipeline"""
    cache = ResourceCache()
    graph = featureset.spec.graph.copy()
    if graph.engine != "sync":
        raise mlrun.errors.MLRunInvalidArgumentError("spark must use sync graph")

    server = create_graph_server(graph=graph, parameters={})
    server.init_states(context=None, namespace=namespace, resource_cache=cache)
    server.init_object(namespace)
    server.context.spark = spark
    event = MockEvent(body=df)
    return server.run(event, get_body=True)


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


def _add_data_steps(
    graph, cache, featureset, targets, source, return_df=False, context=None
):
    _, default_final_step, _ = graph.check_and_process_graph(allow_empty=True)
    validate_target_list(targets=targets)
    validate_target_placement(graph, default_final_step, targets)
    cache.cache_resource(featureset.uri, featureset, True)
    table = add_target_steps(
        graph, featureset, targets, to_df=return_df, final_step=default_final_step
    )
    if table:
        cache.cache_table(featureset.uri, table, True)

    entity_columns = list(featureset.spec.entities.keys())
    key_fields = entity_columns if entity_columns else None

    if source is not None:
        source = get_source_step(
            source,
            key_fields=key_fields,
            time_field=featureset.spec.timestamp_key,
            context=context,
        )
    graph.set_flow_source(source)


def run_ingestion_job(name, featureset, run_config, schedule=None, spark_service=None):
    name = name or f"{featureset.metadata.name}_ingest"
    use_spark = featureset.spec.engine == "spark"
    if use_spark and not run_config.local and not spark_service:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Remote spark ingestion requires the spark service name to be provided"
        )

    default_kind = RuntimeKinds.remotespark if use_spark else RuntimeKinds.job
    spark_runtimes = [RuntimeKinds.remotespark]  # may support spark operator in future

    if not run_config.function:
        function_ref = featureset.spec.function.copy()
        if function_ref.is_empty():
            function_ref = FunctionReference(name=name, kind=default_kind)
        if not function_ref.url:
            function_ref.code = (function_ref.code or "") + _default_job_handler
        run_config.function = function_ref
        run_config.handler = "handler"

    image = None if use_spark else mlrun.mlconf.feature_store.default_job_image
    function = run_config.to_function(default_kind, image)
    if use_spark and function.kind not in spark_runtimes:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "ingest with spark engine require spark function kind"
        )

    function.metadata.project = featureset.metadata.project
    function.metadata.name = function.metadata.name or name

    if not use_spark and not function.spec.image:
        raise mlrun.errors.MLRunInvalidArgumentError("function image must be specified")

    if use_spark and not run_config.local:
        function.with_spark_service(spark_service=spark_service)

    task = mlrun.new_task(
        name=name,
        params=run_config.parameters,
        handler=run_config.handler,
        out_path=featureset.spec.output_path,
    )
    task.spec.secret_sources = run_config.secret_sources
    task.set_label("job-type", "feature-ingest").set_label(
        "feature-set", featureset.uri
    )

    # set run UID and save in the feature set status (linking the features et to the job)
    task.metadata.uid = uuid.uuid4().hex
    featureset.status.run_uri = task.metadata.uid
    featureset.save()

    run = function.run(
        task, schedule=schedule, local=run_config.local, watch=run_config.watch
    )
    if run_config.watch:
        featureset.reload()
    return run


_default_job_handler = """
from mlrun.feature_store.api import ingest
def handler(context):
    ingest(mlrun_context=context)
"""
