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

import mlrun
from mlrun import code_to_function, get_run_db, mlconf, new_project, new_task
from mlrun.utils.vault import VaultStore
from tests.conftest import examples_path, out_path, verify_state

# Set a proper token value for Vault test
user_token = ""


# Set test secrets and configurations - you may need to modify these.
def _set_vault_mlrun_configuration(api_server_port=None):
    if api_server_port:
        mlconf.dbpath = f"http://localhost:{api_server_port}"
    mlconf.secret_stores.vault.url = "http://localhost:8200"
    mlconf.secret_stores.vault.user_token = user_token


# Verify that local activation of Vault functionality is successful. This does not
# test the API-server implementation, which is verified in other tests
@pytest.mark.skipif(user_token == "", reason="no vault configuration")
def test_direct_vault_usage():

    _set_vault_mlrun_configuration()
    project_name = "the-blair-witch-project"

    vault = VaultStore()
    vault.delete_vault_secrets(project=project_name)
    secrets = vault.get_secrets(None, project=project_name)
    assert len(secrets) == 0, "Secrets were not deleted"

    expected_secrets = {"secret1": "123456", "secret2": "654321"}
    vault.add_vault_secrets(expected_secrets, project=project_name)

    secrets = vault.get_secrets(None, project=project_name)
    assert (
        secrets == expected_secrets
    ), "Vault contains different set of secrets than expected"

    secrets = vault.get_secrets(["secret1"], project=project_name)
    assert len(secrets) == 1 and secrets["secret1"] == expected_secrets["secret1"]

    # Test the same thing for user
    user_name = "pikachu"
    vault.delete_vault_secrets(user=user_name)
    secrets = vault.get_secrets(None, user=user_name)
    assert len(secrets) == 0, "Secrets were not deleted"

    vault.add_vault_secrets(expected_secrets, user=user_name)
    secrets = vault.get_secrets(None, user=user_name)
    assert (
        secrets == expected_secrets
    ), "Vault contains different set of secrets than expected"

    # Cleanup
    vault.delete_vault_secrets(project=project_name)
    vault.delete_vault_secrets(user=user_name)


@pytest.mark.skipif(user_token == "", reason="no vault configuration")
def test_vault_end_to_end():
    # This requires an MLRun API server to run and work with Vault. This port should
    # be configured to allow access to the server.
    api_server_port = 57764

    _set_vault_mlrun_configuration(api_server_port)
    project_name = "abc"
    func_name = "vault-function"
    aws_key_value = "1234567890"
    github_key_value = "proj1Key!!!"

    project = new_project(project_name)
    # This call will initialize Vault infrastructure and add the given secrets
    # It executes on the API server
    project.set_secrets(
        {"aws_key": aws_key_value, "github_key": github_key_value},
        provider=mlrun.api.schemas.SecretProviderName.vault,
    )

    # This API executes on the client side
    project_secrets = project.get_vault_secret_keys()
    assert project_secrets == ["aws_key", "github_key"], "secrets not created"

    # Create function and set container configuration
    function = code_to_function(
        name=func_name,
        filename=f"{examples_path}/vault_function.py",
        handler="vault_func",
        project=project_name,
        kind="job",
    )

    function.spec.image = "saarcoiguazio/mlrun:unstable"

    # Create context for the execution
    spec = new_task(
        project=project_name,
        name="vault_test_run",
        handler="vault_func",
        out_path=out_path,
        params={"secrets": ["password", "path", "github_key", "aws_key"]},
    )
    spec.with_secrets("vault", [])

    result = function.run(spec)
    verify_state(result)

    db = get_run_db().connect()
    state, log = db.get_log(result.metadata.uid, project=project_name)
    log = str(log)
    print(state)

    assert (
        log.find(f"value: {aws_key_value}") != -1
    ), "secret value not detected in function output"
    assert (
        log.find(f"value: {github_key_value}") != -1
    ), "secret value not detected in function output"
