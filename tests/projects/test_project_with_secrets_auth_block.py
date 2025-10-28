import pytest

import mlrun


def _auth_prefix() -> str:
    # Matches how the code builds the pattern: format(hashed_access_key="")
    return mlrun.mlconf.secret_stores.kubernetes.auth_secret_name.format(
        hashed_access_key=""
    )


class _StubAzureVaultStore:
    """No-op stub to avoid real Azure config and network."""

    def __init__(self, name: str):
        self.name = name

    def get_secrets(self, secrets):
        # Return empty dict; we only care that client-side validation didn't block.
        return {}


def test_with_secrets_azure_vault_blocks_auth_secret_name(tmp_path, monkeypatch):
    # Ensure we never touch the real Azure implementation
    monkeypatch.setattr(mlrun.secrets, "AzureVaultStore", _StubAzureVaultStore)

    project = mlrun.new_project("proj-auth-block", context=str(tmp_path), save=False)

    forbidden = _auth_prefix() + "anything"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
        project.with_secrets(
            "azure_vault",
            {
                "name": "vault1",
                "k8s_secret": forbidden,
                "tenant_id": "t",
                "vault_url": "https://x",
                "secrets": [],  # required by secrets store path
            },
        )
    assert "Forbidden secret" in str(exc.value)
    assert forbidden in str(exc.value)


def test_with_secrets_azure_vault_allows_non_auth_secret(tmp_path, monkeypatch):
    # Stub Azure so we don't require tenant/client_id config
    monkeypatch.setattr(mlrun.secrets, "AzureVaultStore", _StubAzureVaultStore)

    project = mlrun.new_project("proj-auth-allow", context=str(tmp_path), save=False)

    allowed = "my-regular-k8s-secret"
    # Should not raise
    project.with_secrets(
        "azure_vault",
        {
            "name": "vault1",
            "k8s_secret": allowed,
            "tenant_id": "t",
            "vault_url": "https://x",
            "secrets": [],  # minimal valid payload for add_source
        },
    )
