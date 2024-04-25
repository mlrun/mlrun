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

import os
from time import sleep

import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState
from databricks_wrapper import credentials_path

from mlrun.errors import MLRunRuntimeError


def main():
    if not os.path.exists(credentials_path):
        print("credentials_path not found. Retrying in 10 seconds...")
        sleep(10)
        if not os.path.exists(credentials_path):
            raise MLRunRuntimeError(
                "The Databricks credentials path does not exist."
                " Please manually cancel the job from the Databricks environment."
            )
    with open(credentials_path) as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
    # use for flat yaml only
    for key, value in loaded_data.items():
        os.environ[key] = str(value)
    workspace = WorkspaceClient()
    task_id = os.environ.get("TASK_RUN_ID")
    if os.environ.get("IS_FINISHED", "false").lower() == "false":
        run = workspace.jobs.cancel_run(run_id=task_id).result()
        life_cycle_state = run.as_dict().get("state").get("life_cycle_state")
        if (
            # TERMINATED is also the life_cycle_state of tasks that have already either failed or succeeded
            life_cycle_state
            not in [RunLifeCycleState.TERMINATING, RunLifeCycleState.TERMINATED]
        ):
            raise MLRunRuntimeError(
                f"Cancelling task {task_id} has failed, life cycle state is: {life_cycle_state}. "
                f"Please check the status of this task in the Databricks environment."
            )
        print(f"Cancelling task {task_id} has succeeded.")


main()
