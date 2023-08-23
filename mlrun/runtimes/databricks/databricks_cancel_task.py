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
import yaml
from mlrun.errors import MLRunRuntimeError
from databricks.sdk import WorkspaceClient
from databricks_wrapper import credentials_path


def main():
    with open(credentials_path, "r") as yaml_file:
        loaded_data = yaml.safe_load(yaml_file)
    for key, value in loaded_data.items():
        os.environ[key] = str(value)
    workspace = WorkspaceClient()
    task_id = os.environ.get("TASK_RUN_ID")
    run = workspace.jobs.cancel_run(run_id=task_id).result()
    result_state = run.as_dict().get('state').get('result_state')
    if result_state != "CANCELED":
        raise MLRunRuntimeError(f"canceling task {task_id} has failed."
                                f" Please check the status of this task in the Databricks environment.")


if "__name__" == "__main__":
    main()
