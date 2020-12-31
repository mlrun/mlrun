import pytest
from tests.conftest import (
    examples_path,
    out_path,
    verify_state,
)
from mlrun import new_task, get_run_db, mlconf, code_to_function, new_project
from os import environ

# Uncomment and set proper values for Vault test (at least one is required)
environ["MLRUN_VAULT_ROLE"] = "user:saarc"
environ["MLRUN_VAULT_TOKEN"] = "s.w7orlAPxaWnvf9c815ZDRlcN"


def _has_vault():
    return "MLRUN_VAULT_ROLE" in environ or "MLRUN_VAULT_TOKEN" in environ


# Set test secrets and configurations - you may need to modify these.
def _set_vault_mlrun_configuration(api_server_port=None):
    if api_server_port:
        mlconf.dbpath = f"http://localhost:{api_server_port}"
    mlconf.vault_url = "http://localhost:8200"


# Verify that local activation of Vault functionality is successful. This does not
# test the API-server implementation, which is verified in other tests
@pytest.mark.skipif(not _has_vault(), reason="no vault configuration")
def test_direct_vault_usage():
    from mlrun.utils.vault import VaultStore

    _set_vault_mlrun_configuration()
    project_name = 'the-blair-witch-project'

    vault = VaultStore()
    vault.delete_vault_secrets(project=project_name)
    secrets = vault.get_secrets(None, project=project_name)
    assert len(secrets) == 0, "Secrets were not deleted"

    expected_secrets = {"secret1": "123456", "secret2": "654321"}
    vault.add_vault_secrets(expected_secrets, project=project_name)

    secrets = vault.get_secrets(None, project=project_name)
    assert secrets == expected_secrets, "Vault contains different set of secrets than expected"

    secrets = vault.get_secrets(["secret1"], project=project_name)
    assert len(secrets) == 1 and secrets["secret1"] == expected_secrets["secret1"]

    # Test the same thing for user
    user_name = 'pikachu'
    vault.delete_vault_secrets(user=user_name)
    secrets = vault.get_secrets(None, user=user_name)
    assert len(secrets) == 0, "Secrets were not deleted"

    vault.add_vault_secrets(expected_secrets, user=user_name)
    secrets = vault.get_secrets(None, user=user_name)
    assert secrets == expected_secrets, "Vault contains different set of secrets than expected"

    # Cleanup
    vault.delete_vault_secrets(project=project_name)
    vault.delete_vault_secrets(user=user_name)


@pytest.mark.skipif(not _has_vault(), reason="no vault configuration")
def test_vault_end_to_end():
    api_server_port = 10000

    _set_vault_mlrun_configuration(api_server_port)
    project_name = "abc"
    func_name = "vault-function"
    aws_key_value = "1234567890"
    github_key_value = "proj1Key!!!"

    project = new_project(project_name)
    # This call will initialize Vault infrastructure and add the given secrets
    # It executes on the API server
    project.create_vault_secrets(
        {"aws_key": aws_key_value, "github_key": github_key_value}
    )

    # This API executes on the client side
    project_secrets = project.get_vault_secret_keys()
    assert project_secrets == ["aws_key", "github_key"], "secrets not created"

    # Create function and set container configuration
    function = code_to_function(
        name=func_name,
        filename="{}/vault_function.py".format(examples_path),
        handler="vault_func",
        project=project_name,
        kind="job",
    )
    function.spec.build.base_image = 'saarcoiguazio/mlrun:unstable'
    function.spec.build.image = ".mlrun-vault-image"
    # function.spec.image = ".mlrun-vault-image"
    function.deploy()

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
        log.find("value: {}".format(aws_key_value)) != -1
    ), "secret value not detected in function output"
    assert (
        log.find("value: {}".format(github_key_value)) != -1
    ), "secret value not detected in function output"
