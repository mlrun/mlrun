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
