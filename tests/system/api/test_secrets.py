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
import pathlib
from http import HTTPStatus

import deepdiff
import pytest

import mlrun.api.schemas
import mlrun.errors
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestStam(TestMLRunSystem):
    def test_nothing(self):
        pass

    def test_fail_nothing(self):
        raise Exception("test_fail_nothing")

    def _skip_set_environment(self):
        return True


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

        # Negative test - try to list_project_secrets for k8s secrets (not implemented)
        with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
            self._run_db.list_project_secrets(self.project_name, provider="kubernetes")

        # Negative test - try to create_secret with invalid key
        with pytest.raises(mlrun.errors.MLRunBadRequestError):
            self._run_db.create_project_secrets(
                self.project_name, "kubernetes", {"invalid/key": "value"}
            )

        # Negative test - try to create_secret with forbidden (internal) key
        with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
            self._run_db.create_project_secrets(
                self.project_name, "kubernetes", {"mlrun.key": "value"}
            )

    def test_k8s_project_secrets_with_runtime(self):
        secrets = {"secret1": "JustMySecret", "secret2": "!@#$$%^^&&"}

        # Setup k8s secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        self._run_db.create_project_secrets(self.project_name, "kubernetes", secrets)

        # Run a function using k8s secrets
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        function = mlrun.code_to_function(
            name="test-func",
            project=self.project_name,
            filename=filename,
            handler="secret_test_function",
            kind="job",
            image="mlrun/mlrun",
        )

        # Try running without using with_secrets at all, using the auto-add feature
        task = mlrun.new_task()
        run = function.run(task, params={"secrets": list(secrets.keys())})
        for key, value in secrets.items():
            assert run.outputs[key] == value

        # Test running with an empty list of secrets
        task = mlrun.new_task().with_secrets("kubernetes", [])
        run = function.run(task, params={"secrets": list(secrets.keys())})
        for key, value in secrets.items():
            assert run.outputs[key] == value

        # And with actual secret keys
        task = mlrun.new_task().with_secrets("kubernetes", list(secrets.keys()))
        run = function.run(task, params={"secrets": list(secrets.keys())})
        for key, value in secrets.items():
            assert run.outputs[key] == value

        # Verify that when running with a partial list of secrets, only these secrets are available
        task = mlrun.new_task().with_secrets("kubernetes", ["secret1"])
        run = function.run(task, params={"secrets": list(secrets.keys())})
        expected = {"secret1": secrets["secret1"], "secret2": "None"}
        for key, value in expected.items():
            assert run.outputs[key] == value

        # Cleanup secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
