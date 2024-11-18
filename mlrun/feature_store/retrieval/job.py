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
#
import uuid
from typing import Optional

import mlrun
import mlrun.common.constants as mlrun_constants
from mlrun.config import config as mlconf
from mlrun.model import DataTargetBase, new_task
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.utils import logger

from ...runtimes import RuntimeKinds
from ..common import RunConfig
from .base import BaseMerger


def run_merge_job(
    vector,
    target: DataTargetBase,
    merger: BaseMerger,
    engine: str,
    engine_args: dict,
    spark_service: Optional[str] = None,
    entity_rows=None,
    entity_timestamp_column=None,
    run_config=None,
    drop_columns=None,
    with_indexes=None,
    query=None,
    order_by=None,
    start_time=None,
    end_time=None,
    timestamp_for_filtering=None,
    additional_filters=None,
):
    name = vector.metadata.name
    if not target or not hasattr(target, "to_dict"):
        raise mlrun.errors.MLRunInvalidArgumentError("target object must be specified")
    name = f"{name}-merger"
    run_config = run_config or RunConfig()
    kind = run_config.kind or ("spark" if engine == "spark" else "job")
    run_config.kind = kind
    default_code = _default_merger_handler.replace("{{{engine}}}", merger.__name__)
    if not run_config.function:
        function_ref = vector.spec.function.copy()
        if function_ref.is_empty():
            function_ref = FunctionReference(name=name, kind=kind)
        if not function_ref.url:
            function_ref.code = default_code
        run_config.function = function_ref

    function = run_config.to_function(kind, merger.get_default_image(kind))

    # Avoid overriding a handler that was provided by the user
    if not run_config.handler:
        function.with_code(body=default_code)
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "get_offline_features does not support run_config with a handler"
        )

    function.metadata.project = vector.metadata.project
    function.metadata.name = function.metadata.name or name

    if run_config.kind == RuntimeKinds.remotespark:
        if not spark_service:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "spark_service must be set when running with a remote spark runtime"
            )
        function.with_spark_service(spark_service=spark_service)
    elif run_config.kind == RuntimeKinds.spark:
        if mlconf.is_running_on_iguazio():
            function.with_igz_spark()

        def set_default_resources(resources, setter_function):
            requests = resources.get("requests")
            set_memory = requests is None or "memory" not in requests
            set_cpu = requests is None or "cpu" not in requests
            if set_memory or set_cpu:
                mem = "1G" if set_memory else None
                cpu = "1" if set_cpu else None
                setter_function(mem=mem, cpu=cpu, patch=True)

        set_default_resources(
            function.spec.driver_resources, function.with_driver_requests
        )
        set_default_resources(
            function.spec.executor_resources, function.with_executor_requests
        )
    if start_time and not isinstance(start_time, str):
        start_time = start_time.isoformat()
    if end_time and not isinstance(end_time, str):
        end_time = end_time.isoformat()

    task = new_task(
        name=name,
        params={
            "vector_uri": vector.uri,
            "target": target.to_dict(),
            "entity_timestamp_column": entity_timestamp_column,
            "drop_columns": drop_columns,
            "with_indexes": with_indexes,
            "query": query,
            "order_by": order_by,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_for_filtering": timestamp_for_filtering,
            "engine_args": engine_args,
            "additional_filters": additional_filters,
        },
        inputs={"entity_rows": entity_rows} if entity_rows is not None else {},
    )
    task.spec.secret_sources = run_config.secret_sources
    task.set_label(
        mlrun_constants.MLRunInternalLabels.job_type, "feature-merge"
    ).set_label(mlrun_constants.MLRunInternalLabels.feature_vector, vector.uri)
    task.metadata.uid = uuid.uuid4().hex
    vector.status.run_uri = task.metadata.uid
    vector.save()

    run = function.run(
        task,
        handler=run_config.handler or "merge_handler",
        local=run_config.local,
        watch=run_config.watch,
    )
    logger.info(f"Feature vector merge job started, run id = {run.uid()}")
    return RemoteVectorResponse(vector, run, with_indexes, drop_columns)


class RemoteVectorResponse:
    """get_offline_features response object"""

    def __init__(self, vector, run, with_indexes=False, drop_columns=None):
        self.run = run
        self.vector = vector
        self.with_indexes = with_indexes or self.vector.spec.with_indexes
        self.drop_columns = drop_columns

    @property
    def status(self):
        """vector prep job status (ready, running, error)"""
        return self.run.state()

    def _is_ready(self):
        if self.status != "completed":
            raise mlrun.errors.MLRunTaskNotReadyError(
                "feature vector dataset is not ready"
            )
        self.vector.reload()

    def to_dataframe(self, columns=None, df_module=None, **kwargs):
        """return result as a dataframe object (generated from the dataitem).

        :param columns:   optional, list of columns to select
        :param df_module: optional, py module used to create the DataFrame (e.g. pd, dd, cudf, ..)
        :param kwargs:    extended DataItem.as_df() args
        """
        self._is_ready()
        if not columns:
            columns = list(self.vector.status.features.keys())
            if self.with_indexes:
                columns += self.vector.status.index_keys
                if self.vector.status.timestamp_key is not None:
                    columns.insert(0, self.vector.status.timestamp_key)
            if self.drop_columns:
                for drop_col in self.drop_columns:
                    if drop_col in columns:
                        columns.remove(drop_col)

        file_format = kwargs.get("format")
        if not file_format:
            file_format = self.run.status.results["target"]["kind"]

        df = mlrun.get_dataitem(self.target_uri).as_df(
            columns=columns, df_module=df_module, format=file_format, **kwargs
        )
        if self.with_indexes:
            df.set_index(self.vector.status.index_keys, inplace=True, drop=True)
        return df

    @property
    def target_uri(self):
        """return path of the results file"""
        self._is_ready()
        return self.run.output("target")["path"]


_default_merger_handler = """
import mlrun
import mlrun.feature_store.retrieval
from mlrun.datastore.targets import get_target_driver
def merge_handler(context, vector_uri, target, entity_rows=None, 
                  entity_timestamp_column=None, drop_columns=None, with_indexes=None, query=None,
                  engine_args=None, order_by=None, start_time=None, end_time=None, timestamp_for_filtering=None,
                  additional_filters=None):
    vector = context.get_store_resource(vector_uri)
    store_target = get_target_driver(target, vector)
    if entity_rows:
        entity_rows = entity_rows.as_df()

    context.logger.info(f"Starting vector merge task to {vector.uri}")
    merger = mlrun.feature_store.retrieval.{{{engine}}}(vector, **(engine_args or {}))
    merger.start(entity_rows, entity_timestamp_column, store_target, drop_columns, with_indexes=with_indexes, 
                 query=query, order_by=order_by, start_time=start_time, end_time=end_time,
                 timestamp_for_filtering=timestamp_for_filtering, additional_filters=additional_filters)

    target = vector.status.targets[store_target.name].to_dict()
    context.log_result('feature_vector', vector.uri)
    context.log_result('target', target)
"""  # noqa
