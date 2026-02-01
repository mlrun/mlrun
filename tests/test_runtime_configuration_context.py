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

import pytest

import mlrun
import mlrun.common.constants
import mlrun.runtime_configuration_context


class TestRuntimeConfigurationContext:
    """Tests for core RuntimeConfigurationContext context manager functionality."""

    def test_context_manager_enter_exit(self):
        """Test basic context manager enter/exit."""
        ctx = mlrun.RuntimeConfigurationContext(auth_token_name="test")

        # Before entering
        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

        # Enter
        ctx.__enter__()
        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is ctx
        )

        # Exit
        ctx.__exit__(None, None, None)
        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

    def test_context_manager_nested(self):
        """Test nested context managers properly save and restore."""
        outer_ctx = mlrun.RuntimeConfigurationContext(auth_token_name="outer")
        inner_ctx = mlrun.RuntimeConfigurationContext(auth_token_name="inner")

        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

        with outer_ctx:
            assert (
                mlrun.runtime_configuration_context.runtime_configuration_context.get()
                is outer_ctx
            )

            with inner_ctx:
                assert (
                    mlrun.runtime_configuration_context.runtime_configuration_context.get()
                    is inner_ctx
                )

            # After inner exits, outer is restored
            assert (
                mlrun.runtime_configuration_context.runtime_configuration_context.get()
                is outer_ctx
            )

        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

    def test_context_manager_cleanup_on_exception(self):
        """Test context manager cleans up on exception."""
        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

        with pytest.raises(ValueError):
            with mlrun.RuntimeConfigurationContext(auth_token_name="test"):
                assert (
                    mlrun.runtime_configuration_context.runtime_configuration_context.get()
                    is not None
                )
                raise ValueError("test error")

        # Context should be cleaned up after exception
        assert (
            mlrun.runtime_configuration_context.runtime_configuration_context.get()
            is None
        )

    def test_repr(self):
        """Test __repr__ method."""
        ctx = mlrun.RuntimeConfigurationContext(auth_token_name="my-token")
        assert repr(ctx) == "RuntimeConfigurationContext(auth_token_name='my-token')"

        ctx_none = mlrun.RuntimeConfigurationContext()
        assert repr(ctx_none) == "RuntimeConfigurationContext(auth_token_name=None)"


class TestAuthTokenName:
    """Tests for auth_token_name feature of RuntimeConfigurationContext."""

    def test_get_auth_token_name_within_context(self):
        """Test getting auth token name within context."""
        assert (
            mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
            is None
        )

        with mlrun.RuntimeConfigurationContext(auth_token_name="test-token"):
            assert (
                mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
                == "test-token"
            )

        assert (
            mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
            is None
        )

    def test_get_auth_token_name_nested_contexts(self):
        """Test auth token name with nested contexts."""
        with mlrun.RuntimeConfigurationContext(auth_token_name="outer-token"):
            assert (
                mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
                == "outer-token"
            )

            with mlrun.RuntimeConfigurationContext(auth_token_name="inner-token"):
                assert (
                    mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
                    == "inner-token"
                )

            # After inner exits, outer value is restored
            assert (
                mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
                == "outer-token"
            )

    def test_get_auth_token_name_no_token_set(self):
        """Test get_auth_token_name when context is active but no token set."""
        with mlrun.RuntimeConfigurationContext():
            assert (
                mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
                is None
            )

    def test_env_var_fallback_when_no_context(self, monkeypatch):
        """Test that env var is used when no context manager is active."""
        monkeypatch.setenv(
            mlrun.common.constants.MLRUN_WORKFLOW_RUNNER_AUTH_TOKEN_NAME_ENV_VAR,
            "env-token",
        )

        result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
        assert result == "env-token"

    def test_context_takes_precedence_over_env_var(self, monkeypatch):
        """Test that context var takes precedence over env var."""
        monkeypatch.setenv(
            mlrun.common.constants.MLRUN_WORKFLOW_RUNNER_AUTH_TOKEN_NAME_ENV_VAR,
            "env-token",
        )

        with mlrun.RuntimeConfigurationContext(auth_token_name="context-token"):
            result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
            assert result == "context-token"

        # After context exits, falls back to env var
        result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
        assert result == "env-token"

    def test_returns_none_when_neither_context_nor_env_var_set(self, monkeypatch):
        """Test that None is returned when neither context nor env var is set."""
        monkeypatch.delenv(
            mlrun.common.constants.MLRUN_WORKFLOW_RUNNER_AUTH_TOKEN_NAME_ENV_VAR,
            raising=False,
        )

        result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
        assert result is None
