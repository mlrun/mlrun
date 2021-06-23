from http import HTTPStatus

import deepdiff
import pytest

import mlrun.api.schemas
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestKubernetesProjectSecrets(TestMLRunSystem):
    project_name = "db-system-test-project"

    def test_k8s_project_secrets_using_api(self):
        secrets = {"secret1": "value1", "secret2": "value2"}
        data = {"provider": "kubernetes", "secrets": secrets}
        expected_results = {
            "provider": "kubernetes",
            "secret_keys": [key for key in secrets],
        }

        self._run_db.api_call(
            "DELETE", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )

        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secret-keys?provider=kubernetes"
        )
        assert (
            deepdiff.DeepDiff(
                response.json(), {"provider": "kubernetes", "secret_keys": []}
            )
            == {}
        )

        response = self._run_db.api_call(
            "POST", f"projects/{self.project_name}/secrets", json=data
        )
        assert response.status_code == HTTPStatus.CREATED.value

        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secret-keys?provider=kubernetes"
        )
        assert deepdiff.DeepDiff(response.json(), expected_results) == {}

        # Add a secret key
        add_secret_data = {
            "provider": "kubernetes",
            "secrets": {"secret3": "mySecret!!!"},
        }
        response = self._run_db.api_call(
            "POST", f"projects/{self.project_name}/secrets", json=add_secret_data
        )
        assert response.status_code == HTTPStatus.CREATED.value

        expected_results["secret_keys"].append("secret3")
        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secret-keys?provider=kubernetes"
        )
        assert deepdiff.DeepDiff(response.json(), expected_results) == {}

        # Delete a single secret
        response = self._run_db.api_call(
            "DELETE",
            f"projects/{self.project_name}/secrets?provider=kubernetes&secret=secret1",
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value

        expected_results["secret_keys"].remove("secret1")
        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secret-keys?provider=kubernetes"
        )
        assert deepdiff.DeepDiff(response.json(), expected_results) == {}

        # Cleanup
        response = self._run_db.api_call(
            "DELETE", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value

    def test_k8s_project_secrets_using_httpdb(self):
        secrets = {"secret1": "value1", "secret2": "value2"}
        expected_results = mlrun.api.schemas.SecretKeysData(
            provider="kubernetes", secret_keys=list(secrets.keys())
        )

        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

        response = self._run_db.list_project_secret_keys(
            self.project_name, provider="kubernetes"
        )
        assert response.secret_keys == []

        self._run_db.create_project_secrets(self.project_name, "kubernetes", secrets)

        response = self._run_db.list_project_secret_keys(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.dict(), expected_results.dict()) == {}

        # Add a secret key
        added_secret = {"secret3": "mySecret!!!"}
        self._run_db.create_project_secrets(
            self.project_name, "kubernetes", added_secret
        )

        expected_results.secret_keys.append("secret3")
        response = self._run_db.list_project_secret_keys(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.dict(), expected_results.dict()) == {}

        # Delete secrets
        self._run_db.delete_project_secrets(
            self.project_name, provider="kubernetes", secrets=["secret1", "secret2"]
        )
        expected_results.secret_keys.remove("secret1")
        expected_results.secret_keys.remove("secret2")
        response = self._run_db.list_project_secret_keys(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.dict(), expected_results.dict()) == {}

        # Cleanup
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

        # Negative test - try to list_secrets for k8s secrets (not implemented)
        with pytest.raises(mlrun.errors.MLRunBadRequestError):
            self._run_db.list_project_secrets(self.project_name, provider="kubernetes")
