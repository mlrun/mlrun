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
    add_target_states,
    get_target_driver,
    validate_target_placement,
)

from ..data_types import InferOptions
from ..datastore.store_resources import ResourceCache
from ..runtimes import RuntimeKinds
from ..runtimes.function_reference import FunctionReference
from ..serving.server import create_graph_server


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
    validate_target_placement(graph, default_final_state, targets)
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


def run_ingestion_job(name, featureset, run_config, schedule=None):
    name = name or f"{featureset.metadata.name}_ingest"

    if not run_config.function:
        function_ref = featureset.spec.function.copy()
        if function_ref.is_empty():
            function_ref = FunctionReference(name=name, kind=RuntimeKinds.job)
        if not function_ref.url:
            code = function_ref.code or ""
            if run_config.kind == RuntimeKinds.remotespark:
                function_ref.code = code + _default_spark_handler
            else:
                function_ref.code = code + _default_job_handler
        run_config.function = function_ref
        run_config.handler = "handler"

    image = (
        _default_spark_image()
        if run_config.kind == RuntimeKinds.remotespark
        else mlrun.mlconf.feature_store.default_job_image
    )
    function = run_config.to_function("job", image)
    function.metadata.project = featureset.metadata.project
    function.metadata.name = function.metadata.name or name

    if not function.spec.image:
        raise mlrun.errors.MLRunInvalidArgumentError("function image must be specified")

    task = mlrun.new_task(
        name=name, params=run_config.parameters, handler=run_config.handler
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
from mlrun.feature_store.api import ingest
def handler(context):
    ingest(mlrun_context=context, spark_context=True)
"""
