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
import yaml
from databricks.sdk import WorkspaceClient

MUST_HAVE_VARIABLES = ["DATABRICKS_TOKEN", "DATABRICKS_HOST"]
MLRUN_ROOT_DIR = "/mlrun_tests"


def is_dbfs_configured(config_file_path):
    with config_file_path.open() as fp:
        config = yaml.safe_load(fp)
    env_params = config["env"]
    for necessary_variable in MUST_HAVE_VARIABLES:
        if env_params.get(necessary_variable, None) is None:
            return False
    return True


def setup_dbfs_dirs(workspace: WorkspaceClient, specific_test_class_dir: str, subdirs: list):
    all_paths = [file_info.path for file_info in workspace.dbfs.list("/")]
    if MLRUN_ROOT_DIR not in all_paths:
        workspace.dbfs.mkdirs(MLRUN_ROOT_DIR)
        return
    specific_test_class_path = f"{MLRUN_ROOT_DIR}{specific_test_class_dir}"
    mlrun_test_dirs = [file_info.path for file_info in workspace.dbfs.list(MLRUN_ROOT_DIR)]
    if specific_test_class_path in mlrun_test_dirs:
        workspace.dbfs.delete(specific_test_class_path, recursive=True)
    for test_dir in subdirs:
        workspace.dbfs.mkdirs(f"{specific_test_class_path}{test_dir}")


def teardown_dbfs_dirs(workspace: WorkspaceClient, specific_test_class_dir: str):
    specific_test_class_path = f"{MLRUN_ROOT_DIR}{specific_test_class_dir}"
    all_paths_under_class_path = [
        file_info.path for file_info in workspace.dbfs.list(specific_test_class_path)
    ]
    for path in all_paths_under_class_path:
        workspace.dbfs.delete(path, recursive=True)
