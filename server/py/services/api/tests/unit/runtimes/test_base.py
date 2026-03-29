# Copyright 2026 Iguazio
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

from unittest.mock import MagicMock, patch

import kubernetes.client as k8s_client

import mlrun.runtimes
import mlrun.secrets
from mlrun.config import config as mlconf

from services.api.runtime_handlers.base import BaseRuntimeHandler


def test_user_env_var_not_overridden_by_global_secret(monkeypatch):
    """User-set AWS_ACCESS_KEY_ID is preserved when global secret has the same key.

    Exercises the full add_k8s_secrets_to_spec flow with a mocked k8s helper,
    verifying that the plain-value guard prevents the override.
    """

    runtime = mlrun.runtimes.KubejobRuntime()
    env_var_name = "AWS_ACCESS_KEY_ID"
    user_value = "user-custom-value"

    # User sets a plain env var before secret injection
    runtime.set_env(env_var_name, user_value)

    global_secret_name = "minio-credentials"
    global_secrets = {
        "AWS_ACCESS_KEY_ID": "minio-key-id",
        "AWS_SECRET_ACCESS_KEY": "minio-secret-key",
    }

    # Mock k8s helper to return global secrets
    mock_k8s = MagicMock()
    mock_k8s.get_secret_data.return_value = global_secrets

    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "global_function_env_secret_name",
        global_secret_name,
    )
    with patch(
        "services.api.runtime_handlers.base.framework.utils.singletons.k8s.get_k8s_helper",
        return_value=mock_k8s,
    ):
        BaseRuntimeHandler.add_k8s_secrets_to_spec(
            secrets=None,
            runtime=runtime,
            project_name=None,
            encode_key_names=True,
        )

    # The user's plain value must be preserved
    env_var = find_env_var(runtime, env_var_name)
    assert env_var is not None, f"Env var '{env_var_name}' not found in runtime spec"
    assert env_var.value == user_value, (
        f"REGRESSION: User's env var '{env_var_name}' was overridden "
        f"by global secret. Expected value='{user_value}', "
        f"got value={env_var.value!r}, value_from={env_var.value_from!r}."
    )

    # AWS_SECRET_ACCESS_KEY was NOT set by user, so it should come from global secret
    other_env = find_env_var(runtime, "AWS_SECRET_ACCESS_KEY")
    assert other_env is not None, (
        "AWS_SECRET_ACCESS_KEY should be injected from global secret"
    )
    assert other_env.value_from is not None, (
        "AWS_SECRET_ACCESS_KEY should be a secretKeyRef (not user-set)"
    )


def test_user_env_var_not_overridden_by_project_secret(monkeypatch):
    """User-set env var is preserved when project secret has the same key.

    Uses encode_key_names=False (nuclio path) for simplicity, so env var
    names match secret key names directly.
    """

    runtime = mlrun.runtimes.KubejobRuntime()
    secret_key = "MY_PROJECT_SECRET"
    user_value = "user-custom-project-value"

    # User sets the env var before secret injection
    runtime.set_env(secret_key, user_value)

    project_name = "test-project"
    project_secret_keys = ["MY_PROJECT_SECRET", "OTHER_SECRET"]

    # Mock k8s helper
    mock_k8s = MagicMock()
    mock_k8s.get_secret_data.return_value = {}  # no global secrets
    mock_k8s.get_project_secret_name.return_value = "project-secret-name"
    mock_k8s.get_project_secret_keys.return_value = project_secret_keys

    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "global_function_env_secret_name",
        "",
    )
    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "auto_add_project_secrets",
        True,
    )
    with patch(
        "services.api.runtime_handlers.base.framework.utils.singletons.k8s.get_k8s_helper",
        return_value=mock_k8s,
    ):
        BaseRuntimeHandler.add_k8s_secrets_to_spec(
            secrets=None,
            runtime=runtime,
            project_name=project_name,
            encode_key_names=False,
        )

    # User's value must be preserved
    env_var = find_env_var(runtime, secret_key)
    assert env_var is not None
    assert env_var.value == user_value, (
        f"REGRESSION: User's env var '{secret_key}' was overridden "
        f"by project secret. Expected value='{user_value}', "
        f"got value={env_var.value!r}, value_from={env_var.value_from!r}."
    )

    # OTHER_SECRET was NOT set by user, so it should come from project secret
    other_env = find_env_var(runtime, "OTHER_SECRET")
    assert other_env is not None, "OTHER_SECRET should be injected from project secret"
    assert other_env.value_from is not None, (
        "OTHER_SECRET should be a secretKeyRef (not user-set)"
    )


def test_project_secret_overrides_global_secret_for_same_key(monkeypatch):
    """When both global and project secrets share a key, project must win."""

    runtime = mlrun.runtimes.KubejobRuntime()

    # Shared key in both global and project secrets, user did NOT set it
    shared_key = "SHARED_SECRET"
    global_secret_name = "global-secret"
    global_secrets = {shared_key: "global-value"}
    project_name = "test-project"
    project_secret_keys = [shared_key]

    mock_k8s = MagicMock()
    mock_k8s.get_secret_data.return_value = global_secrets
    mock_k8s.get_project_secret_name.return_value = "project-secret-name"
    mock_k8s.get_project_secret_keys.return_value = project_secret_keys

    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "global_function_env_secret_name",
        global_secret_name,
    )
    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "auto_add_project_secrets",
        True,
    )
    with patch(
        "services.api.runtime_handlers.base.framework.utils.singletons.k8s.get_k8s_helper",
        return_value=mock_k8s,
    ):
        BaseRuntimeHandler.add_k8s_secrets_to_spec(
            secrets=None,
            runtime=runtime,
            project_name=project_name,
            encode_key_names=False,
        )

    # The env var should come from the PROJECT secret, not the global one
    env_var = find_env_var(runtime, shared_key)
    assert env_var is not None, f"Env var '{shared_key}' not found in runtime spec"
    assert env_var.value_from is not None, (
        f"'{shared_key}' should be a secretKeyRef, not a plain value"
    )
    secret_ref = env_var.value_from.secret_key_ref
    assert secret_ref.name == "project-secret-name", (
        f"Project > global priority broken: '{shared_key}' should reference "
        f"project secret 'project-secret-name', but references '{secret_ref.name}'"
    )


def test_user_plain_var_wins_over_both_global_and_project_secrets(monkeypatch):
    """When user sets a plain env var, neither global nor project secret overrides it."""

    runtime = mlrun.runtimes.KubejobRuntime()
    shared_key = "SHARED_KEY"
    user_value = "user-wins"

    runtime.set_env(shared_key, user_value)

    global_secret_name = "global-secret"
    global_secrets = {shared_key: "global-value"}
    project_name = "test-project"
    project_secret_keys = [shared_key]

    mock_k8s = MagicMock()
    mock_k8s.get_secret_data.return_value = global_secrets
    mock_k8s.get_project_secret_name.return_value = "project-secret-name"
    mock_k8s.get_project_secret_keys.return_value = project_secret_keys

    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "global_function_env_secret_name",
        global_secret_name,
    )
    monkeypatch.setattr(
        mlconf.secret_stores.kubernetes,
        "auto_add_project_secrets",
        True,
    )
    with patch(
        "services.api.runtime_handlers.base.framework.utils.singletons.k8s.get_k8s_helper",
        return_value=mock_k8s,
    ):
        BaseRuntimeHandler.add_k8s_secrets_to_spec(
            secrets=None,
            runtime=runtime,
            project_name=project_name,
            encode_key_names=False,
        )

    # User's plain value must survive both global and project injection
    env_var = find_env_var(runtime, shared_key)
    assert env_var is not None
    assert env_var.value == user_value, (
        f"User's plain env var '{shared_key}' was overridden. "
        f"Expected value='{user_value}', "
        f"got value={env_var.value!r}, value_from={env_var.value_from!r}."
    )


def find_env_var(runtime, name):
    """Find an env var by name in the runtime spec."""

    for env_var in runtime.spec.env:
        if isinstance(env_var, k8s_client.V1EnvVar):
            if env_var.name == name:
                return env_var
        elif isinstance(env_var, dict):
            if env_var.get("name") == name:
                return env_var
    return None
