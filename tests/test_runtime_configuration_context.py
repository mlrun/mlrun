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

    def test_config_not_used_when_no_context(self, monkeypatch):
        """Test that config value is not used - only context manager matters."""
        monkeypatch.setattr(
            mlrun.mlconf.auth_with_oauth_token,
            "token_name",
            "config-token",
        )

        # Even with config set, returns None because no context manager is active
        result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
        assert result is None

    def test_context_only_source_of_token(self, monkeypatch):
        """Test that only context manager provides the token, not config."""
        monkeypatch.setattr(
            mlrun.mlconf.auth_with_oauth_token,
            "token_name",
            "config-token",
        )

        with mlrun.RuntimeConfigurationContext(auth_token_name="context-token"):
            result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
            assert result == "context-token"

        # After context exits, returns None (not config value)
        result = mlrun.runtime_configuration_context.RuntimeConfigurationContext.get_auth_token_name()
        assert result is None
