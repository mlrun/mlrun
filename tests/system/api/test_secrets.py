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
import os
import pathlib
import re
import time
import uuid
from http import HTTPStatus

import deepdiff
import igz_mgmt
import pytest

import mlrun.common.schemas
import mlrun.errors
from mlrun.config import config
from tests.system.base import TestMLRunSystem

# This is a copy from server/py/services/api/utils/events/iguazio.py because the system tests simulate user env
# where this module is not available
PROJECT_SECRET_CREATED = "Security.Project.Secret.Created"
PROJECT_SECRET_UPDATED = "Security.Project.Secret.Updated"
PROJECT_SECRET_DELETED = "Security.Project.Secret.Deleted"


@TestMLRunSystem.skip_test_if_env_not_configured
class TestKubernetesProjectSecrets(TestMLRunSystem):
    project_name = "db-system-test-project"

    @pytest.mark.enterprise
    def test_audit_project_secret_events(self):
        secret_key = str(uuid.uuid4())
        secrets = {secret_key: "JustMySecret"}

        # ensure no project secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

        # create secret
        now = datetime.datetime.now(datetime.timezone.utc)
        self.project.set_secrets(secrets=secrets)

        self._ensure_audit_events(
            PROJECT_SECRET_CREATED,
            now,
            "secret_keys",
            secret_key,
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        another_secret_key = str(uuid.uuid4())
        secrets.update({another_secret_key: "one"})
        self.project.set_secrets(secrets=secrets)
        self._ensure_audit_events(
            PROJECT_SECRET_UPDATED,
            now,
            "secret_keys",
            another_secret_key,
        )

        # delete secrets
        now = datetime.datetime.now(datetime.timezone.utc)
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        self._ensure_audit_events(
            PROJECT_SECRET_DELETED,
            now,
            "project_name",
            self.project_name,
        )

    @pytest.mark.enterprise
    def test_delete_project_secret_events(self):
        """
        Test flow:
            1. Delete project secrets of project with no secrets - should not emit event
            2. Create 2 secrets - should emit created event
            3. Delete 1 secret - should emit update event
            4. Delete all secrets - should emit deleted event
            5. Delete project - should not emit secret deleted event
        """
        secret_key1 = str(uuid.uuid4())
        secret_key2 = str(uuid.uuid4())
        secrets = {
            secret_key1: "JustMySecret",
            secret_key2: "MyOtherSecret",
        }

        # ensure no project secrets
        start = datetime.datetime.now(datetime.timezone.utc)
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        time.sleep(1)
        audit_events = igz_mgmt.AuditEvent.list(
            self._igz_mgmt_client,
            filter_by={
                "source": "mlrun-api",
                "kind": PROJECT_SECRET_DELETED,
                "timestamp_iso8601": f"[$ge]{start.isoformat()}",
            },
        )
        assert len(audit_events) == 0

        now = datetime.datetime.now(datetime.timezone.utc)
        self.project.set_secrets(secrets=secrets)
        self._ensure_audit_events(
            PROJECT_SECRET_CREATED,
            now,
            "project_name",
            self.project_name,
        )

        # delete 1 of the secrets
        now = datetime.datetime.now(datetime.timezone.utc)
        self._run_db.delete_project_secrets(
            self.project_name, provider="kubernetes", secrets=[secret_key1]
        )

        # project secret should remain (updated)
        self._ensure_audit_events(
            PROJECT_SECRET_UPDATED,
            now,
            "secret_keys",
            secret_key1,
        )

        # delete all secrets
        now = datetime.datetime.now(datetime.timezone.utc)
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        self._ensure_audit_events(
            PROJECT_SECRET_DELETED,
            now,
            "project_name",
            self.project_name,
        )

        # delete the secret-less project
        now = datetime.datetime.now(datetime.timezone.utc)
        self._run_db.delete_project(
            self.project_name, mlrun.common.schemas.DeletionStrategy.cascade
        )

        # should not emit deleted event
        time.sleep(1)
        audit_events = igz_mgmt.AuditEvent.list(
            self._igz_mgmt_client,
            filter_by={
                "source": "mlrun-api",
                "kind": PROJECT_SECRET_DELETED,
                "timestamp_iso8601": f"[$ge]{now.isoformat()}",
            },
        )
        assert len(audit_events) == 0

        # assert 1 deleted event from the start of the test
        audit_events = igz_mgmt.AuditEvent.list(
            self._igz_mgmt_client,
            filter_by={
                "source": "mlrun-api",
                "kind": PROJECT_SECRET_DELETED,
                "timestamp_iso8601": f"[$ge]{start.isoformat()}",
            },
        )
        assert len(audit_events) == 1

    @pytest.mark.smoke
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
        # This test validates both retrieval flows:
        # 1. Secrets accessed via get_secret_or_env
        # 2. Secrets accessed directly from os.environ when use_prefix=False.

        secrets = {"secret1": "JustMySecret", "SECRET2": "!@#$$%^^&&"}
        no_prefix_secrets = {"secret3": "ShhItsASecret", "SECRET4": "AnotherSecret"}

        # Setup k8s secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")
        self._run_db.create_project_secrets(self.project_name, "kubernetes", secrets)

        # Function setup
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        function = mlrun.code_to_function(
            name="test-func",
            project=self.project_name,
            filename=filename,
            handler="secret_test_function",
            kind="job",
            image="artifactory.iguazeng.com:10557/yaelg/mlrun/mlrun:1.8.0-rc21",
        )

        test_cases = [
            {
                # Run without explicitly passing any secrets. Validate the auto-add feature
                "task": mlrun.new_task(),
                "params": list(secrets.keys()),
                "expected": secrets,
            },
            {
                # Run with an empty list of secrets. Validate that all secrets are still auto-added
                "task": mlrun.new_task().with_secrets("kubernetes", []),
                "params": list(secrets.keys()),
                "expected": secrets,
            },
            {
                # Run with all secret keys explicitly passed. Validate that the correct values are retrieved
                "task": mlrun.new_task().with_secrets(
                    "kubernetes", list(secrets.keys())
                ),
                "params": list(secrets.keys()),
                "expected": secrets,
            },
            {
                # Run with only a partial list of secret keys. Validate that only specified secrets are accessible
                "task": mlrun.new_task().with_secrets("kubernetes", ["secret1"]),
                "params": list(secrets.keys()),
                "expected": {"secret1": secrets["secret1"], "SECRET2": "None"},
            },
        ]

        for case in test_cases:
            run = function.run(
                case["task"],
                params={"secrets": case["params"]},
            )
            for key, value in case["expected"].items():
                assert run.outputs[key] == value

        # Validate that the correct values are retrieved directly from environment variables
        self._run_db.create_project_secrets(
            self.project_name, "kubernetes", no_prefix_secrets
        )
        run = function.run(
            mlrun.new_task().with_secrets("kubernetes", list(no_prefix_secrets.keys())),
            params={"secrets": list(no_prefix_secrets.keys()), "use_prefix": False},
        )
        for key, value in no_prefix_secrets.items():
            assert run.outputs[key] == value

        # Cleanup secrets
        self._run_db.delete_project_secrets(self.project_name, provider="kubernetes")

    @pytest.mark.enterprise
    @pytest.mark.parametrize(
        "operation",
        [
            "deploy",
            "run",
            "save",
        ],
    )
    def test_masked_access_key(self, operation):
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        function = mlrun.code_to_function(
            name="test-masked-access-key",
            project=self.project_name,
            filename=filename,
            handler="access_key_verifier",
            kind="job",
            image="mlrun/mlrun",
        )

        # generate mlrun auth session
        function.metadata.credentials.access_key = (
            mlrun.model.Credentials.generate_access_key
        )

        # set v3io credentials
        v3io_access_key = os.environ.get("V3IO_ACCESS_KEY")
        function.set_env(name="V3IO_ACCESS_KEY", value=v3io_access_key)
        function.set_env(name="V3IO_USERNAME", value=os.environ.get("V3IO_USERNAME"))

        if operation == "deploy":
            function.deploy()
        elif operation == "run":
            function.run(params={"v3io_access_key": v3io_access_key})
        elif operation == "save":
            function.save(versioned=False)
        else:
            assert False, f"Bad operation {operation}"

        run_db = mlrun.get_run_db()
        runtime = run_db.get_function(function.metadata.name, self.project_name)
        function = mlrun.new_function(runtime=runtime)

        # verify v3io access key was masked
        masked_v3io_access_key = function.get_env("V3IO_ACCESS_KEY")
        secret_name_regex = config.secret_stores.kubernetes.auth_secret_name.format(
            hashed_access_key=".+"
        )
        assert re.match(
            secret_name_regex,
            masked_v3io_access_key["secretKeyRef"]["name"],
        )

        # auth session should be generated and masked
        masked_mlrun_session = function.get_env("MLRUN_AUTH_SESSION")
        assert re.match(secret_name_regex, masked_mlrun_session["secretKeyRef"]["name"])

        if operation != "run":
            function.run(params={"v3io_access_key": v3io_access_key})

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
    ) -> list[igz_mgmt.AuditEvent]:
        def _get_audit_events():
            self._logger.info(
                "Trying to get audit events",
                event_kind=event_kind,
                since_time=since_time.isoformat(),
            )
            audit_events = igz_mgmt.AuditEvent.list(
                self._igz_mgmt_client,
                filter_by={
                    "source": "mlrun-api",
                    "kind": event_kind,
                    "timestamp_iso8601": f"[$ge]{since_time.isoformat()}",
                },
            )
            assert len(audit_events) > 0
            return audit_events

        # wait for 30 seconds for the audit events to be available
        return mlrun.utils.retry_until_successful(
            3,
            10 * 3,
            self._logger,
            True,
            _get_audit_events,
        )
