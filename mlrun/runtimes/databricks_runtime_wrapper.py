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


def run_mlrun_databricks_job(
    context,
    internal_handler,
    internal_code,
    token_key="DATABRICKS_TOKEN",
    timeout=20,
    **kwargs,
):
    import datetime
    import json
    import uuid

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.compute import ClusterSpec
    from databricks.sdk.service.jobs import Run, SparkPythonTask, SubmitTask

    import mlrun

    logger = context.logger

    def upload_file(workspace: WorkspaceClient, script_path_on_dbfs: str, handler):
        modified_code = internal_code
        with workspace.dbfs.open(script_path_on_dbfs, write=True, overwrite=True) as f:
            f.write(modified_code.encode("UTF8"))

    workspace = WorkspaceClient(token=mlrun.get_secret_or_env(key=token_key))

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    script_path_on_dbfs = (
        f"/home/{workspace.current_user.me().user_name}/mlrun_databricks_runtime/"
        f"sample_{formatted_date_time}_{uuid.uuid4()}.py"
    )

    upload_file(
        workspace=workspace,
        script_path_on_dbfs=script_path_on_dbfs,
        handler=internal_handler,
    )

    def print_status(run: Run):
        statuses = [f"{t.task_key}: {t.state.life_cycle_state}" for t in run.tasks]
        logger.info(f'workflow intermediate status: {", ".join(statuses)}')

    try:
        cluster_id = mlrun.get_secret_or_env("DATABRICKS_CLUSTER_ID")
        if cluster_id:
            logger.info(f"run with exists cluster_id: {cluster_id}")
            waiter = workspace.jobs.submit(
                run_name=f"py-sdk-run-{formatted_date_time}",
                tasks=[
                    SubmitTask(
                        task_key=f"hello_world-{formatted_date_time}",
                        existing_cluster_id=cluster_id,
                        spark_python_task=SparkPythonTask(
                            python_file=f"dbfs:{script_path_on_dbfs}",
                            parameters=[json.dumps(kwargs)],
                        ),
                    )
                ],
            )
        else:
            logger.info("run with new cluster_id")
            waiter = workspace.jobs.submit(
                run_name=f"py-sdk-run-{formatted_date_time}",
                tasks=[
                    SubmitTask(
                        task_key=f"hello_world-{formatted_date_time}",
                        new_cluster=ClusterSpec(
                            spark_version=workspace.clusters.select_spark_version(
                                long_term_support=True
                            ),
                            node_type_id=workspace.clusters.select_node_type(
                                local_disk=True
                            ),
                            num_workers=1,
                        ),
                        spark_python_task=SparkPythonTask(
                            python_file=f"dbfs:{script_path_on_dbfs}",
                            parameters=[json.dumps(kwargs)],
                        ),
                    )
                ],
            )
        logger.info(f"starting to poll: {waiter.run_id}")
        run = waiter.result(
            timeout=datetime.timedelta(minutes=timeout), callback=print_status
        )

        run_output = workspace.jobs.get_run_output(run.tasks[0].run_id)
        context.log_result("databricks_runtime task:", run_output)
    finally:
        workspace.dbfs.delete(script_path_on_dbfs)

    logger.info(f"job finished: {run.run_page_url}")
    logger.info(f"logs:\n{run_output.logs}")
