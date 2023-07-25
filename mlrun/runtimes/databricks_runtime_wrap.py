def run_mlrun_databricks_job(context, internal_handler, is_local_code=False, token_key="DATABRICKS_TOKEN", **kwargs):
    import mlrun
    from databricks.sdk.service.jobs import Run, SubmitTask, SparkPythonTask
    from databricks.sdk.service.compute import ClusterSpec
    from databricks.sdk import WorkspaceClient
    import uuid
    import datetime
    import json
    import shutil
    import os
    import tempfile
    import ast
    import astor

    project_name = context.project
    project = mlrun.get_or_create_project(project_name)
    get_secret_func = project.get_secret
    logger = context.logger

    def get_modified_code(function_name, script_file, handler):
        with open(script_file, 'r') as file:
            tree = ast.parse(file.read())

        # Find the function node to delete
        to_delete = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                to_delete = node
                break

        if to_delete:
            # Remove the function node
            tree.body.remove(to_delete)

        # Generate the modified code
        modified_code = astor.to_source(tree)
        if handler:
            handler_function = \
                f"""
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('handler_arguments', help='')
handler_arguments = parser.parse_args().handler_arguments
handler_arguments = json.loads(handler_arguments)
{handler}(**handler_arguments)
"""
            modified_code += handler_function
        return modified_code

    def copy_current_file(dst_dir):
        current_script = os.path.abspath(__file__)
        # Create a temporary directory and copy the file inside the context
        # Generate the temporary file path
        temp_file_path = os.path.join(dst_dir, os.path.basename(current_script))
        # Copy the current script to the temporary location
        shutil.copy(current_script, temp_file_path)
        return temp_file_path

    def upload_file(workspace: WorkspaceClient, script_path_on_dbfs: str, handler):
        with tempfile.TemporaryDirectory(prefix='databricks_runtime_scripts_') as temp_dir:
            temp_file_path = copy_current_file(temp_dir)
            modified_code = get_modified_code(function_name='run_mlrun_databricks_job', script_file=temp_file_path,
                                              handler=handler)
            with workspace.dbfs.open(script_path_on_dbfs, write=True, overwrite=True) as f:
                f.write(modified_code.encode("UTF8"))

    workspace = WorkspaceClient(token=mlrun.get_secret_or_env(key=token_key, secret_provider=get_secret_func))

    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    script_path_on_dbfs = f"/home/{workspace.current_user.me().user_name}/mlrun_databricks_runtime/sample_{formatted_date_time}_{uuid.uuid4()}.py"  # todo return this

    if is_local_code:
        upload_file(workspace=workspace, script_path_on_dbfs=script_path_on_dbfs,
                    handler=internal_handler)

    def print_status(run: Run):
        statuses = [f"{t.task_key}: {t.state.life_cycle_state}" for t in run.tasks]
        logger.info(f'workflow intermediate status: {", ".join(statuses)}')

    try:
        cluster_id = os.environ.get('CLUSTER_ID')
        if cluster_id:
            waiter = workspace.jobs.submit(
                run_name=f"py-sdk-run-{formatted_date_time}",
                tasks=[
                    SubmitTask(
                        task_key=f"hello_world-{formatted_date_time}",
                        existing_cluster_id=cluster_id,
                        spark_python_task=SparkPythonTask(python_file=f"dbfs:{script_path_on_dbfs}",
                                                          parameters=[json.dumps(kwargs)]),
                    )
                ],
            )
        else:
            waiter = workspace.jobs.submit(
                run_name=f"py-sdk-run-{formatted_date_time}",
                tasks=[
                    SubmitTask(
                        task_key=f"hello_world-{formatted_date_time}",
                        new_cluster=ClusterSpec(
                            spark_version=workspace.clusters.select_spark_version(
                                long_term_support=True
                            ),
                            node_type_id=workspace.clusters.select_node_type(local_disk=True),
                            num_workers=1,
                        ),
                        spark_python_task=SparkPythonTask(python_file=f"dbfs:{script_path_on_dbfs}",
                                                          parameters=[json.dumps(kwargs)]),
                    )
                ],
            )
        logger.info(f"starting to poll: {waiter.run_id}")
        run = waiter.result(timeout=datetime.timedelta(minutes=15), callback=print_status)

        run_output = workspace.jobs.get_run_output(run.tasks[0].run_id)
        logger.info(f"run_output: {run_output}")
    finally:
        workspace.dbfs.delete(script_path_on_dbfs)

    logger.info(f"job finished: {run.run_page_url}")
    logger.info(f"logs:\n{run_output.logs}")
