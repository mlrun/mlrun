import uuid

import mlrun
from mlrun.model import new_task
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.utils import logger

from ..common import RunConfig


def run_merge_job(
    vector,
    target,
    entity_rows=None,
    timestamp_column=None,
    run_config=None,
    drop_columns=None,
    with_indexes=None,
):
    name = vector.metadata.name
    if not target or not hasattr(target, "to_dict"):
        raise mlrun.errors.MLRunInvalidArgumentError("target object must be specified")
    name = f"{name}_merger"
    run_config = run_config or RunConfig()
    if not run_config.function:
        function_ref = vector.spec.function.copy()
        if function_ref.is_empty():
            function_ref = FunctionReference(name=name, kind="job")
        if not function_ref.url:
            function_ref.code = _default_merger_handler
        run_config.function = function_ref

    function = run_config.to_function(
        "job", mlrun.mlconf.feature_store.default_job_image
    )
    function.metadata.project = vector.metadata.project
    function.metadata.name = function.metadata.name or name
    task = new_task(
        name=name,
        params={
            "vector_uri": vector.uri,
            "target": target.to_dict(),
            "timestamp_column": timestamp_column,
            "drop_columns": drop_columns,
            "with_indexes": with_indexes,
        },
        inputs={"entity_rows": entity_rows},
    )
    task.spec.secret_sources = run_config.secret_sources
    task.set_label("job-type", "feature-merge").set_label("feature-vector", vector.uri)
    task.metadata.uid = uuid.uuid4().hex
    vector.status.run_uri = task.metadata.uid
    vector.save()

    run = function.run(
        task,
        handler=run_config.handler or "merge_handler",
        local=run_config.local,
        watch=run_config.watch,
    )
    logger.info(f"feature vector merge job started, run id = {run.uid()}")
    return RemoteVectorResponse(vector, run)


class RemoteVectorResponse:
    """get_offline_features response object"""

    def __init__(self, vector, run):
        self.run = run
        self.vector = vector

    @property
    def status(self):
        """vector prep job status (ready, running, error)"""
        return self.run.state()

    def _is_ready(self):
        if self.status != "completed":
            raise mlrun.errors.MLRunTaskNotReady("feature vector dataset is not ready")
        self.vector.reload()

    def to_dataframe(self, columns=None, df_module=None, **kwargs):
        """return result as a dataframe object (generated from the dataitem).

        :param columns:   optional, list of columns to select
        :param df_module: optional, py module used to create the DataFrame (e.g. pd, dd, cudf, ..)
        :param kwargs:    extended DataItem.as_df() args
        """
        return mlrun.get_dataitem(self.target_uri).as_df(
            columns=columns, df_module=df_module, **kwargs
        )

    @property
    def target_uri(self):
        """return path of the results file"""
        self._is_ready()
        return self.run.output("target")["path"]


_default_merger_handler = """
import mlrun
from mlrun.feature_store.retrieval import LocalFeatureMerger
from mlrun.datastore.targets import get_target_driver
def merge_handler(context, vector_uri, target, entity_rows=None, 
                  timestamp_column=None, drop_columns=None, with_indexes=None):
    vector = context.get_store_resource(vector_uri)
    store_target = get_target_driver(target, vector)
    entity_timestamp_column = timestamp_column or vector.spec.timestamp_field
    if entity_rows:
        entity_rows = entity_rows.as_df()

    context.logger.info(f"starting vector merge task to {vector.uri}")
    merger = LocalFeatureMerger(vector)
    resp = merger.start(entity_rows, entity_timestamp_column, store_target, drop_columns, with_indexes=with_indexes)
    target = vector.status.targets[store_target.name].to_dict()
    context.log_result('feature_vector', vector.uri)
    context.log_result('target', target)
"""  # noqa
