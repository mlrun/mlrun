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

import mlrun
import mlrun.launcher.client
import mlrun.runtimes


def test_enrich_run_auth_from_function_context_manager_applies_token():
    """Test that context manager token is applied to run.spec.auth."""
    runtime = mlrun.runtimes.KubejobRuntime()
    run = mlrun.run.RunObject()

    # Without context, no auth should be set
    mlrun.launcher.client.ClientBaseLauncher._enrich_run_auth_from_function(
        runtime, run
    )
    assert run.spec.auth is None or "token_name" not in (run.spec.auth or {})

    # With context, token should be applied
    run2 = mlrun.run.RunObject()
    with mlrun.RuntimeConfigurationContext(auth_token_name="context-token"):
        mlrun.launcher.client.ClientBaseLauncher._enrich_run_auth_from_function(
            runtime, run2
        )
    assert run2.spec.auth["token_name"] == "context-token"


def test_enrich_run_auth_from_function_context_manager_overrides_function_auth():
    """Test that context manager token overrides function-level auth."""
    runtime = mlrun.runtimes.KubejobRuntime()
    runtime.spec.auth = {"token_name": "function-token"}
    run = mlrun.run.RunObject()

    with mlrun.RuntimeConfigurationContext(auth_token_name="context-token"):
        mlrun.launcher.client.ClientBaseLauncher._enrich_run_auth_from_function(
            runtime, run
        )

    # Context manager should override
    assert run.spec.auth["token_name"] == "context-token"


def test_enrich_run_auth_from_function_uses_function_auth_when_no_context():
    """Test that function auth is used when no context manager is active."""
    runtime = mlrun.runtimes.KubejobRuntime()
    runtime.spec.auth = {"token_name": "function-token"}
    run = mlrun.run.RunObject()

    mlrun.launcher.client.ClientBaseLauncher._enrich_run_auth_from_function(
        runtime, run
    )

    assert run.spec.auth["token_name"] == "function-token"


def test_enrich_run_auth_from_function_preserves_run_auth():
    """Test that existing run auth is preserved when no context or function auth."""
    runtime = mlrun.runtimes.KubejobRuntime()
    run = mlrun.run.RunObject()
    run.spec.auth = {"token_name": "run-token"}

    mlrun.launcher.client.ClientBaseLauncher._enrich_run_auth_from_function(
        runtime, run
    )

    assert run.spec.auth["token_name"] == "run-token"
