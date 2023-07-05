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
import datetime
import pathlib
import typing
import uuid
from http import HTTPStatus

import deepdiff
import igz_mgmt
import pytest

import mlrun.api.utils.events.iguazio
import mlrun.common.schemas
import mlrun.errors
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestKubernetesProjectSecrets(TestMLRunSystem):
    project_name = "db-system-test-project"

    @pytest.mark.enterprise
    def test_audit_secret(self):
        secret_key = str(uuid.uuid4())
        secrets = {secret_key: "JustMySecret"}

        # ensure no project secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

        # create secret
        now = datetime.datetime.utcnow()
        project = self._run_db.get_project(self.project_name)
        project.set_secrets(secrets=secrets)

        self._ensure_audit_events(
            mlrun.api.utils.events.iguazio.PROJECT_SECRET_CREATED,
            now,
            "secret_keys",
            secret_key,
        )

        now = datetime.datetime.utcnow()
        another_secret_key = str(uuid.uuid4())
        secrets.update({another_secret_key: "one"})
        project.set_secrets(secrets=secrets)
        self._ensure_audit_events(
            mlrun.api.utils.events.iguazio.PROJECT_SECRET_UPDATED,
            now,
            "secret_keys",
            another_secret_key,
        )

        # delete secrets
        now = datetime.datetime.utcnow()
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        self._ensure_audit_events(
            mlrun.api.utils.events.iguazio.PROJECT_SECRET_DELETED,
            now,
            "project_name",
            self.project_name,
        )

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
        expected_results = mlrun.common.schemas.SecretKeysData(
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

    def _ensure_audit_events(
        self,
        event_kind: str,
        since_time: datetime.datetime,
        parameter_text_name: str,
        parameter_text_value: str,
    ):
        actual_event = None
        for event in self._get_audit_events(event_kind, since_time):
            if not event.parameters_text:
                continue
            for parameter_text in event.parameters_text:
                if (
                    parameter_text.name == parameter_text_name
                    and parameter_text_value in parameter_text.value
                ):
                    actual_event = event
                    break
        assert actual_event is not None, "Failed to find the audit event"

    def _get_audit_events(
        self, event_kind: str, since_time: datetime.datetime
    ) -> typing.List[igz_mgmt.AuditEvent]:
        def _get_audit_events():
            audit_events = igz_mgmt.AuditEvent.list(
                self._igz_mgmt_client,
                filter_by={
                    "source": "mlrun-api",
                    "kind": event_kind,
                    "timestamp_iso8601": f"[$ge]{since_time.isoformat()}Z",
                },
            )
            assert len(audit_events) > 0
            return audit_events

        return mlrun.utils.retry_until_successful(
            3,
            60 * 3,
            self._logger,
            True,
            _get_audit_events,
        )
