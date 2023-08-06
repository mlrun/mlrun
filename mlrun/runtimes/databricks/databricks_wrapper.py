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

import datetime
import json
import uuid
from base64 import b64decode

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.jobs import Run, SparkPythonTask, SubmitTask

import mlrun


def run_mlrun_databricks_job(
    context,
    mlrun_internal_code,
    mlrun_internal_token_key="DATABRICKS_TOKEN",
    mlrun_internal_timeout_minutes=20,
    mlrun_internal_number_of_workers=1,
    **kwargs,
):

    logger = context.logger
    workspace = WorkspaceClient(
        token=mlrun.get_secret_or_env(key=mlrun_internal_token_key)
    )
    mlrun_databricks_job_id = uuid.uuid4()
    script_path_on_dbfs = (
        f"/home/{workspace.current_user.me().user_name}/mlrun_databricks_runtime/"
        f"mlrun_task_{mlrun_databricks_job_id}.py"
    )

    code = b64decode(mlrun_internal_code).decode("utf-8")
    with workspace.dbfs.open(script_path_on_dbfs, write=True, overwrite=True) as f:
        f.write(code.encode("utf-8"))

    def print_status(run: Run):
        statuses = [f"{t.task_key}: {t.state.life_cycle_state}" for t in run.tasks]
        logger.info(f'workflow intermediate status: {", ".join(statuses)}')

    try:
        cluster_id = mlrun.get_secret_or_env("DATABRICKS_CLUSTER_ID")
        submit_task_kwargs = {}
        if cluster_id:
            logger.info(f"run with exists cluster_id: {cluster_id}")
            submit_task_kwargs["existing_cluster_id"] = cluster_id
        else:
            logger.info("run with new cluster_id")
            submit_task_kwargs["new_cluster"] = ClusterSpec(
                spark_version=workspace.clusters.select_spark_version(
                    long_term_support=True
                ),
                node_type_id=workspace.clusters.select_node_type(local_disk=True),
                num_workers=mlrun_internal_number_of_workers,
            )
        waiter = workspace.jobs.submit(
            run_name=f"py-sdk-run-{mlrun_databricks_job_id}",
            tasks=[
                SubmitTask(
                    task_key=f"hello_world-{mlrun_databricks_job_id}",
                    spark_python_task=SparkPythonTask(
                        python_file=f"dbfs:{script_path_on_dbfs}",
                        parameters=[json.dumps(kwargs)],
                    ),
                    **submit_task_kwargs,
                )
            ],
        )
        logger.info(f"starting to poll: {waiter.run_id}")
        run = waiter.result(
            timeout=datetime.timedelta(minutes=mlrun_internal_timeout_minutes),
            callback=print_status,
        )

        run_output = workspace.jobs.get_run_output(run.tasks[0].run_id)
        context.log_result("databricks_runtime_task", run_output.as_dict())
    finally:
        workspace.dbfs.delete(script_path_on_dbfs)

    logger.info(f"job finished: {run.run_page_url}")
    logger.info(f"logs:\n{run_output.logs}")
