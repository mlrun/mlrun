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
import typing
import uuid
from base64 import b64decode

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import OperationFailed
from databricks.sdk.service.compute import ClusterSpec
from databricks.sdk.service.jobs import Run, RunTask, SparkPythonTask, SubmitTask

import mlrun
from mlrun.errors import MLRunRuntimeError

credentials_path = "/mlrun/databricks_credentials.yaml"


def get_task(databricks_run: Run) -> RunTask:
    if len(databricks_run.tasks) == 0:
        raise MLRunRuntimeError(
            "Cannot find tasks related to the Databricks job run ID in Databricks environment."
        )
    elif len(databricks_run.tasks) > 1:
        raise MLRunRuntimeError(
            "More than one task related to the Databricks job run ID has been found."
            " Did you manually run any tasks from this job?"
        )
    return databricks_run.tasks[0]


def save_credentials(
    workspace: WorkspaceClient,
    waiter,
    host: str,
    token: str,
    cluster_id: typing.Optional[str],
    is_finished: bool,
):
    databricks_run = workspace.jobs.get_run(run_id=waiter.run_id)
    task_run_id = get_task(databricks_run=databricks_run).run_id
    credentials = {
        "DATABRICKS_HOST": host,
        "DATABRICKS_TOKEN": token,
        "TASK_RUN_ID": task_run_id,
        "IS_FINISHED": is_finished,
    }
    if cluster_id:
        credentials["DATABRICKS_CLUSTER_ID"] = cluster_id

    with open(credentials_path, "w") as yaml_file:
        yaml.dump(credentials, yaml_file, default_flow_style=False)


def run_mlrun_databricks_job(
    context,
    task_parameters: dict,
    **kwargs,
):
    mlrun_databricks_job_id = uuid.uuid4()
    spark_app_code = task_parameters["spark_app_code"]
    token_key = task_parameters.get("token_key", "DATABRICKS_TOKEN")
    databricks_token = mlrun.get_secret_or_env(key=token_key)
    host = mlrun.get_secret_or_env(key="DATABRICKS_HOST")
    timeout_minutes = task_parameters.get("timeout_minutes", 20)
    number_of_workers = task_parameters.get("number_of_workers", 1)
    new_cluster_spec = task_parameters.get("new_cluster_spec")
    databricks_run_name = task_parameters.get(
        "databricks_run_name", f"mlrun_task_{mlrun_databricks_job_id}"
    )

    logger = context.logger
    workspace = WorkspaceClient(token=databricks_token)

    script_path_on_dbfs = (
        f"/home/{workspace.current_user.me().user_name}/mlrun_databricks_runtime/"
        f"{databricks_run_name}.py"
    )

    spark_app_code = b64decode(spark_app_code).decode("utf-8")
    with workspace.dbfs.open(script_path_on_dbfs, write=True, overwrite=True) as f:
        f.write(spark_app_code.encode("utf-8"))

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
            cluster_spec_kwargs = {
                "spark_version": workspace.clusters.select_spark_version(
                    long_term_support=True
                ),
                "node_type_id": workspace.clusters.select_node_type(local_disk=True),
                "num_workers": number_of_workers,
            }
            if new_cluster_spec:
                cluster_spec_kwargs.update(new_cluster_spec)
            submit_task_kwargs["new_cluster"] = ClusterSpec(**cluster_spec_kwargs)
        waiter = workspace.jobs.submit(
            run_name=databricks_run_name,
            tasks=[
                SubmitTask(
                    task_key=databricks_run_name,
                    spark_python_task=SparkPythonTask(
                        python_file=f"dbfs:{script_path_on_dbfs}",
                        parameters=[json.dumps(kwargs)],
                    ),
                    **submit_task_kwargs,
                )
            ],
        )
        logger.info(f"starting to poll: {waiter.run_id}")
        save_credentials(
            workspace=workspace,
            waiter=waiter,
            host=host,
            token=databricks_token,
            cluster_id=cluster_id,
            is_finished=False,
        )
        try:
            run = waiter.result(
                timeout=datetime.timedelta(minutes=timeout_minutes),
                callback=print_status,
            )
        except OperationFailed:
            databricks_run = workspace.jobs.get_run(run_id=waiter.run_id)
            task_run_id = get_task(databricks_run=databricks_run).run_id
            error_dict = workspace.jobs.get_run_output(task_run_id).as_dict()
            error_trace = error_dict.pop("error_trace", "")
            custom_error = "error information and metadata:\n"
            custom_error += json.dumps(error_dict, indent=1)
            custom_error += "\nerror trace from databricks:\n" if error_trace else ""
            custom_error += error_trace
            raise MLRunRuntimeError(custom_error)
        finally:
            save_credentials(
                workspace=workspace,
                waiter=waiter,
                host=host,
                token=databricks_token,
                cluster_id=cluster_id,
                is_finished=True,
            )
        run_output = workspace.jobs.get_run_output(get_task(run).run_id)
        context.log_result("databricks_runtime_task", run_output.as_dict())
    finally:
        workspace.dbfs.delete(script_path_on_dbfs)

    logger.info(f"job finished: {run.run_page_url}")
    logger.info(f"logs:\n{run_output.logs}")
