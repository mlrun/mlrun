from http import HTTPStatus

import deepdiff

from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestKubernetesProjectSecrets(TestMLRunSystem):
    project_name = "db-system-test-project"

    def test_k8s_project_secrets_using_api(self):
        secrets = {"secret1": "value1", "secret2": "value2"}
        data = {"provider": "kubernetes", "secrets": secrets}
        expected_results = {
            "provider": "kubernetes",
            "secrets": {key: None for key in secrets},
        }

        self._run_db.api_call(
            "DELETE", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )

        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )
        assert (
            deepdiff.DeepDiff(
                response.json(), {"provider": "kubernetes", "secrets": {}}
            )
            == {}
        )

        response = self._run_db.api_call(
            "POST", f"projects/{self.project_name}/secrets", json=data
        )
        assert response.status_code == HTTPStatus.CREATED.value

        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secrets?provider=kubernetes"
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

        expected_results["secrets"]["secret3"] = None
        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )
        assert deepdiff.DeepDiff(response.json(), expected_results) == {}

        # Delete a single secret
        response = self._run_db.api_call(
            "DELETE",
            f"projects/{self.project_name}/secrets?provider=kubernetes&secret=secret1",
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value

        expected_results["secrets"].pop("secret1")
        response = self._run_db.api_call(
            "GET", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )
        assert deepdiff.DeepDiff(response.json(), expected_results) == {}

        # Cleanup
        response = self._run_db.api_call(
            "DELETE", f"projects/{self.project_name}/secrets?provider=kubernetes"
        )
        assert response.status_code == HTTPStatus.NO_CONTENT.value

    def test_k8s_project_secrets_using_httpdb(self):
        secrets = {"secret1": "value1", "secret2": "value2"}
        expected_results = {key: None for key in secrets}

        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

        response = self._run_db.get_project_secrets(
            self.project_name, provider="kubernetes"
        )
        assert response.secrets == {}

        self._run_db.create_project_secrets(self.project_name, "kubernetes", secrets)

        response = self._run_db.get_project_secrets(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.secrets, expected_results) == {}

        # Add a secret key
        added_secret = {"secret3": "mySecret!!!"}
        self._run_db.create_project_secrets(
            self.project_name, "kubernetes", added_secret
        )

        expected_results["secret3"] = None
        response = self._run_db.get_project_secrets(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.secrets, expected_results) == {}

        # Delete secrets
        self._run_db.delete_project_secrets(
            self.project_name, provider="kubernetes", secrets=["secret1", "secret2"]
        )
        expected_results.pop("secret1")
        expected_results.pop("secret2")
        response = self._run_db.get_project_secrets(
            self.project_name, provider="kubernetes"
        )
        assert deepdiff.DeepDiff(response.secrets, expected_results) == {}

        # Cleanup
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
