# Copyright 2018 Iguazio
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
import pytest

from mlrun import code_to_function, get_run_db, mlconf, new_task
from tests.conftest import out_path, verify_state

azure_key_vault_k8s_secret = ""
api_db_port = 56295


@pytest.mark.skipif(azure_key_vault_k8s_secret == "", reason="no Azure vault secret")
def test_azure_vault_end_to_end():
    mlconf.dbpath = f"http://localhost:{api_db_port}"

    project_name = "proj1"

    # Create function and set container configuration
    function = code_to_function(
        name="azure_vault_func",
        filename="vault_function.py",
        handler="vault_func",
        project=project_name,
        kind="job",
    )

    function.spec.image = "mlrun/mlrun:unstable"

    # Create context for the execution
    spec = new_task(
        project=project_name,
        name="azure_vault_test_run",
        handler="vault_func",
        out_path=out_path,
        params={"secrets": ["demo-key-1", "demo-key-2"]},
    )
    spec.with_secrets(
        "azure_vault",
        {
            "name": "saar-key-vault",
            "k8s_secret": azure_key_vault_k8s_secret,
            "secrets": [],
        },
    )

    result = function.run(spec)
    verify_state(result)

    db = get_run_db().connect()
    db.get_log(result.metadata.uid, project=project_name)
