import pytest
from tests.conftest import (
    examples_path,
    has_vault,
    out_path,
    verify_state,
)
from mlrun import new_task, get_run_db


@pytest.mark.skipif(not has_vault(), reason="no vault configuration")
def test_vault_secrets():
    from mlrun import mlconf, code_to_function, new_project

    # Set test secrets and configurations - you may need to modify these.
    mlconf.dbpath = "http://localhost:63579"
    mlconf.vault_url = "http://localhost:8200"

    proj_name = "abc"
    func_name = "vault-func"
    aws_key_value = "1234567890"
    github_key_value = "proj1Key!!!"

    proj = new_project(proj_name, use_vault=True)
    proj.create_vault_secrets(
        {"aws_key": aws_key_value, "github_key": github_key_value}
    )
    secs = proj.get_vault_secret_keys()
    assert secs == ["aws_key", "github_key"], "secrets not created"

    proj.save(to_db=True, to_file=False)

    # Create function and set container configuration
    func = code_to_function(
        name=func_name,
        filename="{}/vault_function.py".format(examples_path),
        handler="vault_func",
        project=proj_name,
        kind="job",
    )
    # func.spec.build.base_image = 'saarcoiguazio/mlrun:unstable'
    func.spec.build.image = ".mlrun-vault-image"
    # func.spec.image = ".mlrun-vault-image"
    func.deploy()

    # Create context for the execution
    spec = new_task(
        project=proj_name,
        name="vault_test_run",
        handler="vault_func",
        out_path=out_path,
        params={"secrets": ["password", "path", "github_key", "aws_key"]},
    )
    spec.with_secrets("vault", [])

    result = func.run(spec)
    verify_state(result)

    db = get_run_db().connect()
    state, log = db.get_log(result.metadata.uid, project=proj_name)
    log = str(log)
    print(state)

    assert (
        log.find("value: {}".format(aws_key_value)) != -1
    ), "secret value not detected in function output"
    assert (
        log.find("value: {}".format(github_key_value)) != -1
    ), "secret value not detected in function output"
