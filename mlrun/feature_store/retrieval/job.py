import uuid

import mlrun
from mlrun.model import new_task
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.utils import logger


def run_merge_job(
    vector,
    target,
    entity_rows=None,
    timestamp_column=None,
    local=None,
    watch=None,
    drop_columns=None,
    function=None,
    secrets=None,
    auto_mount=None,
):
    name = vector.metadata.name
    if not name:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "feature vector name must be specified"
        )
    if not target or not hasattr(target, "to_dict"):
        raise mlrun.errors.MLRunInvalidArgumentError("target object must be specified")
    name = f"{name}_merger"
    if not function:
        function_ref = vector.spec.function
        if not function_ref.to_dict():
            function_ref = FunctionReference(name=name, kind="job")
        function_ref.image = (
            function_ref.image or mlrun.mlconf.feature_store.default_job_image
        )
        if not function_ref.url:
            function_ref.code = _default_merger_handler
        function = function_ref.to_function()

    if auto_mount:
        function.apply(mlrun.platforms.auto_mount())

    function.metadata.project = vector.metadata.project
    task = new_task(
        name=name,
        params={
            "vector_uri": vector.uri,
            "target": target.to_dict(),
            "timestamp_column": timestamp_column,
            "drop_columns": drop_columns,
        },
        inputs={"entity_rows": entity_rows},
    )
    if secrets:
        task.with_secrets("inline", secrets)
    task.metadata.uid = uuid.uuid4().hex
    vector.status.run_uri = task.metadata.uid
    vector.save()

    run = function.run(task, handler="merge_handler", local=local, watch=watch)
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

    def to_dataframe(self):
        """return result as dataframe"""
        return mlrun.get_dataitem(self.target_uri).as_df()

    @property
    def target_uri(self):
        """return path of the results file"""
        self._is_ready()
        return self.run.output("target")["path"]


_default_merger_handler = """
import mlrun
from mlrun.feature_store.retrieval import LocalFeatureMerger
from mlrun.datastore.targets import get_target_driver
def merge_handler(context, vector_uri, target, entity_rows=None, timestamp_column=None, drop_columns=None):
    vector = context.get_store_resource(vector_uri)
    store_target = get_target_driver(target, vector)
    entity_timestamp_column = timestamp_column or vector.spec.timestamp_field
    if entity_rows:
        entity_rows = entity_rows.as_df()

    context.logger.info(f"starting vector merge task to {vector.uri}")
    merger = LocalFeatureMerger(vector)
    resp = merger.start(entity_rows, entity_timestamp_column, store_target, drop_columns)
    target = vector.status.targets[store_target.name].to_dict()
    context.log_result('feature_vector', vector.uri)
    context.log_result('target', target)
"""
