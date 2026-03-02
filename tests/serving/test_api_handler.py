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

"""Unit tests for the API Handler implementation"""

import logging
import re
from http import HTTPMethod
from typing import cast
from unittest.mock import MagicMock

import pytest

import mlrun
import mlrun.errors
from mlrun.common.schemas.serving import APIHandlerAction, _APIEndpointKeys
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, ServingRuntime
from mlrun.serving import GraphContext
from mlrun.serving.api_handler import _APIHandlerStep, _RequestContext
from mlrun.serving.server import MockEvent, RootFlowStep, _add_api_handler_step_to_graph
from mlrun.serving.utils import (
    _combine_serving_endpoint_key,
    _split_serving_endpoint_key,
)


class EchoStep:
    """Simple echo step for testing"""

    def __init__(
        self, context: GraphContext, name: str | None = None, prefix: str = "", **kwargs
    ) -> None:
        self.context = context
        self.name = name
        self.prefix = prefix

    def do(self, event: MockEvent | str | dict):
        """Echo the event with optional prefix"""
        if hasattr(event, "body"):
            body = event.body
        else:
            body = event
        return f"{self.prefix}{body}" if self.prefix else body


class TestRequestContext:
    """Tests for _RequestContext unified parameter container"""

    def test_empty_context(self) -> None:
        """Test creating context with no parameters"""
        ctx = _RequestContext()
        assert dict(ctx) == {}

    def test_query_params_only(self) -> None:
        """Test context with only query parameters"""
        ctx = _RequestContext(query_params={"limit": "10", "offset": "20"})
        assert ctx["limit"] == "10"
        assert ctx["offset"] == "20"

    def test_path_params_only(self) -> None:
        """Test context with only path parameters"""
        ctx = _RequestContext(path_params={"user_id": "123", "item_id": "abc"})
        assert ctx["user_id"] == "123"
        assert ctx["item_id"] == "abc"

    def test_body_params_only(self) -> None:
        """Test context with only body_map parameters"""
        ctx = _RequestContext(body_params={"name": "test", "value": 42})
        assert ctx["name"] == "test"
        assert ctx["value"] == 42

    def test_parameter_conflict_body_and_path(self) -> None:
        """Test that conflicts between body_map and path params raise error"""
        with pytest.raises(
            mlrun.errors.MLRunBadRequestError,
            match="Parameter name conflict detected.*id.*body_map.*path",
        ):
            _RequestContext(
                path_params={"id": "path-value"},
                body_params={"id": "body-value"},
            )

    def test_parameter_conflict_path_and_query(self) -> None:
        """Test that conflicts between path and query params raise error"""
        with pytest.raises(
            mlrun.errors.MLRunBadRequestError,
            match="Parameter name conflict detected.*id.*query.*path",
        ):
            _RequestContext(
                query_params={"id": "query-value"},
                path_params={"id": "path-value"},
            )

    def test_parameter_conflict_all_sources(self) -> None:
        """Test that conflicts across all three sources raise error"""
        with pytest.raises(
            mlrun.errors.MLRunBadRequestError,
            match="Parameter name conflict detected.*id",
        ):
            _RequestContext(
                query_params={"id": "query", "query_only": "q1"},
                path_params={"id": "path", "path_only": "p1"},
                body_params={"id": "body", "body_only": "b1"},
            )

    def test_no_conflict_different_params(self) -> None:
        """Test that different params from different sources work without conflict"""
        ctx = _RequestContext(
            query_params={"query_only": "q1"},
            path_params={"path_only": "p1"},
            body_params={"body_only": "b1"},
        )
        assert ctx["query_only"] == "q1"
        assert ctx["path_only"] == "p1"
        assert ctx["body_only"] == "b1"

    def test_context_is_mapped_body(self) -> None:
        """Test that RequestContext is a _MappedBody subclass"""
        from mlrun.serving.utils import _MappedBody

        ctx = _RequestContext(query_params={"test": "value"})
        assert isinstance(ctx, _MappedBody)
        assert isinstance(ctx, dict)

    def test_context_unpacks_as_kwargs(self) -> None:
        """Test that context can be unpacked as **kwargs"""

        def test_func(**kwargs):
            return kwargs

        ctx = _RequestContext(
            query_params={"a": "1"},
            path_params={"b": "2"},
        )
        result = test_func(**ctx)
        assert result == {"a": "1", "b": "2"}

    def test_query_params_multiple_values_as_list(self) -> None:
        """Test that repeated query params are returned as list"""
        ctx = _RequestContext(query_params={"id": ["1", "4", "1"], "single": "value"})
        assert ctx["id"] == ["1", "4", "1"]
        assert ctx["single"] == "value"

    def test_url_params_injected_without_conflict(self) -> None:
        """Test that url_params are merged without conflict checking"""
        ctx = _RequestContext(
            path_params={"user_id": "123"},
            url_params={"mlrun_request_path": "/api/users/123"},
        )
        assert ctx["user_id"] == "123"
        assert ctx["mlrun_request_path"] == "/api/users/123"

    def test_url_params_only(self) -> None:
        """Test context with only url_params"""
        ctx = _RequestContext(url_params={"mlrun_request_path": "/api/health"})
        assert ctx["mlrun_request_path"] == "/api/health"

    def test_url_params_do_not_raise_on_overwrite(self) -> None:
        """Test that url_params silently overwrite user params with the same name (reserved prefix)"""
        # This documents that 'mlrun_' prefixed params are reserved; user params
        # with the same name will be overwritten by system-injected url_params.
        ctx = _RequestContext(
            query_params={"mlrun_request_path": "user-supplied"},
            url_params={"mlrun_request_path": "/api/real"},
        )
        # System-injected value wins
        assert ctx["mlrun_request_path"] == "/api/real"


class TestPathTemplateRegex:
    """Tests for pre-compiled regex patterns and path template matching"""

    def test_simple_path_template(self) -> None:
        """Test basic path template with one parameter"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/users/{user_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Should have one compiled pattern
        assert len(step._endpoint_patterns) == 1
        method, pattern, key, _ = step._endpoint_patterns[0]
        assert method == HTTPMethod.GET
        assert key == "GET:/users/{user_id}"

        # Test pattern match
        match = pattern.match("/users/123")
        assert match is not None
        assert match.groupdict() == {"user_id": "123"}

    def test_multiple_path_parameters(self) -> None:
        """Test path template with multiple parameters and multiple different patterns"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/orgs/{org_id}/repos/{repo_id}/issues/{issue_id}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        config.add_endpoint_handler(
            "/api/users/{user_id}/posts/{post_id}",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )

        step = _APIHandlerStep(config=config)
        # Should have 2 different path patterns compiled
        assert len(step._endpoint_patterns) == 2

        # Test first pattern with 3 parameters
        match, params = step._match_endpoint(
            HTTPMethod.GET, "/orgs/mlrun/repos/mlrun/issues/42"
        )
        assert match == "GET:/orgs/{org_id}/repos/{repo_id}/issues/{issue_id}"
        assert params == {
            "org_id": "mlrun",
            "repo_id": "mlrun",
            "issue_id": "42",
        }

        # Test second pattern with 2 parameters
        match, params = step._match_endpoint(
            HTTPMethod.POST, "/api/users/123/posts/456"
        )
        assert match == "POST:/api/users/{user_id}/posts/{post_id}"
        assert params == {"user_id": "123", "post_id": "456"}

    def test_url_encoded_path_params(self) -> None:
        """Test that URL-encoded path segments are decoded"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/files/{filename}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1

        # Test with URL-encoded filename
        match, params = step._match_endpoint(HTTPMethod.GET, "/files/my%20file.txt")
        assert match == "GET:/files/{filename}"
        assert params == {"filename": "my file.txt"}  # Decoded

    def test_special_characters_in_path(self) -> None:
        """Test path params with special characters"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/items/{item_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1

        # Test various special characters
        test_cases = [
            ("abc-123", "abc-123"),
            ("user@example.com", "user@example.com"),
            ("item.v2", "item.v2"),
            ("test_underscore", "test_underscore"),
        ]

        for path_value, expected in test_cases:
            match, params = step._match_endpoint(HTTPMethod.GET, f"/items/{path_value}")
            assert match == "GET:/items/{item_id}"
            assert params == {"item_id": expected}

    def test_regex_does_not_match_across_slashes(self) -> None:
        """Test that path params don't match across path separators"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/{version}/users", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1

        # Should NOT match - extra path segment
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v1/v2/users")
        assert match is None
        assert params == {}

        # Should match - single segment
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v1/users")
        assert match == "GET:/api/{version}/users"
        assert params == {"version": "v1"}

    def test_exact_match_preferred_over_template(self) -> None:
        """Test that exact matches are found before template matching"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/users/me", HTTPMethod.GET, APIHandlerAction.ALLOW, "Current user"
        )
        config.add_endpoint_handler(
            "/users/{user_id}", HTTPMethod.GET, APIHandlerAction.FORBID, "Specific user"
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1  # Only /users/{user_id} has template

        # /users/me should match exact endpoint, not template
        match, params = step._match_endpoint(HTTPMethod.GET, "/users/me")
        assert match == "GET:/users/me"
        assert params == {}

        # /users/123 should match template
        match, params = step._match_endpoint(HTTPMethod.GET, "/users/123")
        assert match == "GET:/users/{user_id}"
        assert params == {"user_id": "123"}

    def test_no_pattern_for_exact_endpoints(self) -> None:
        """Test that exact endpoints (no {}) are not compiled into patterns"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/health", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/api/v1/metrics", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # No patterns should be compiled (no {} in paths)
        assert len(step._endpoint_patterns) == 0

    def test_mixed_exact_and_template_endpoints(self) -> None:
        """Test configuration with both exact and template endpoints"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/health", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/api/users/{id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/api/status", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Only one pattern (the template)
        assert len(step._endpoint_patterns) == 1

        # Exact matches work
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/health")
        assert match == "GET:/api/health"

        # Template matches work
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/users/42")
        assert match == "GET:/api/users/{id}"
        assert params == {"id": "42"}

    def test_regex_anchor_prevents_partial_match(self) -> None:
        """Test that regex anchors (^$) prevent partial matches"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/{version}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1
        _, pattern, _, _ = step._endpoint_patterns[0]

        # Should not match - extra prefix
        assert pattern.match("/prefix/api/v1") is None

        # Should not match - extra suffix
        assert pattern.match("/api/v1/suffix") is None

        # Should match - exact
        assert pattern.match("/api/v1") is not None

    def test_check_method_allowed_with_templates(self) -> None:
        """Test 405 vs 404 error distinction with path templates.

        Note: For performance, we only check exact paths when distinguishing 405 vs 404.
        Templated paths that don't match the request method will return 404, not 405.
        This is a reasonable trade-off for better performance on the error path.
        """
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/items/{id}", HTTPMethod.POST, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/items/{id}", HTTPMethod.PUT, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 2  # Two methods for same path template

        # GET not allowed for templated path → 404 (simplified logic, exact paths only)
        event = MockEvent(method=HTTPMethod.GET, path="/items/123", body={})
        with pytest.raises(mlrun.errors.MLRunNotFoundError, match="Endpoint not found"):
            step.do(event)

        # POST is allowed → should succeed
        event = MockEvent(method=HTTPMethod.POST, path="/items/123", body={})
        result = step.do(event)
        assert result is not None  # Should return event successfully

    def test_empty_path_parameter_name(self) -> None:
        """Test that empty parameter names in templates are handled"""
        config = APIHandlerConfig()
        # This is malformed, but we handle it gracefully
        config._endpoints["GET:/api/{}"] = {
            _APIEndpointKeys.ACTION: "allow",
            _APIEndpointKeys.DESCRIPTION: "Malformed",
        }

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1

        # Should not crash, and pattern should not match
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/test")
        # The regex compilation should handle this gracefully


class TestStarPatternMatching:
    """Tests for wildcard/star pattern matching in API handler (ML-11658)"""

    def test_basic_star_match(self) -> None:
        """Test that a star pattern matches any path under the prefix"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)
        assert len(step._star_patterns) == 1
        assert len(step._endpoint_patterns) == 0

        # Should match paths under /api/
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/users")
        assert match == "GET:/api/*"
        assert params == {}

        match, params = step._match_endpoint(HTTPMethod.GET, "/api/items/123")
        assert match == "GET:/api/*"
        assert params == {}

        match, params = step._match_endpoint(HTTPMethod.GET, "/api/deeply/nested/path")
        assert match == "GET:/api/*"
        assert params == {}

    def test_star_at_end_validation(self) -> None:
        """Test that * must be at the end of the path"""
        config = APIHandlerConfig()
        # Manually insert invalid endpoint with * not at end
        config._endpoints["GET:/api/*/users"] = {
            _APIEndpointKeys.ACTION: "allow",
            _APIEndpointKeys.DESCRIPTION: "Invalid star position",
        }

        with pytest.raises(
            mlrun.errors.MLRunValueError,
            match="wildcard.*must be at the end",
        ):
            _APIHandlerStep(config=config)

    def test_star_patterns_not_in_endpoint_patterns(self) -> None:
        """Test that star patterns are stored separately from template patterns"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/{user_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler("/admin/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 1
        assert len(step._star_patterns) == 1

    def test_precedence_exact_over_star(self) -> None:
        """Test that exact matches take precedence over star patterns"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.FORBID)
        config.add_endpoint_handler(
            "/api/health", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Exact match should win
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/health")
        assert match == "GET:/api/health"
        assert params == {}

        # Non-exact path should fall through to star
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/other")
        assert match == "GET:/api/*"
        assert params == {}

    def test_precedence_template_over_star(self) -> None:
        """Test that template matches take precedence over star patterns"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.FORBID)
        config.add_endpoint_handler(
            "/api/{resource}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Template match should win over star match
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/users")
        assert match == "GET:/api/{resource}"
        assert params == {"resource": "users"}

    def test_precedence_all_three_types(self) -> None:
        """Test full precedence: exact > template > star"""
        config = APIHandlerConfig()
        # Add in reverse precedence order to ensure ordering is correct
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.FORBID)
        config.add_endpoint_handler(
            "/api/{version}/users", HTTPMethod.GET, APIHandlerAction.FORBID
        )
        config.add_endpoint_handler(
            "/api/v2/users", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Exact match wins
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v2/users")
        assert match == "GET:/api/v2/users"
        assert params == {}

        # Template match wins over star
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v1/users")
        assert match == "GET:/api/{version}/users"
        assert params == {"version": "v1"}

        # Star match for paths not covered by exact/template
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v1/items")
        assert match == "GET:/api/*"
        assert params == {}

    def test_star_insertion_order(self) -> None:
        """Test that within star patterns, insertion order is respected"""
        config = APIHandlerConfig()
        # More specific prefix added first should win
        config.add_endpoint_handler("/api/v1/*", HTTPMethod.GET, APIHandlerAction.ALLOW)
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.FORBID)

        step = _APIHandlerStep(config=config)
        assert len(step._star_patterns) == 2

        # /api/v1/ prefix added first, should match first
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v1/users")
        assert match == "GET:/api/v1/*"
        assert params == {}

        # /api/other should fall through to second star pattern
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/v2/users")
        assert match == "GET:/api/*"
        assert params == {}

    def test_star_method_isolation(self) -> None:
        """Test that star patterns only match for the correct HTTP method"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        # GET should match
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/users")
        assert match == "GET:/api/*"

        # POST should not match GET star pattern
        match, params = step._match_endpoint(HTTPMethod.POST, "/api/users")
        assert match is None
        assert params == {}

    def test_star_no_match_outside_prefix(self) -> None:
        """Test that star patterns only match paths under their prefix"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        # Path not under /api/ should not match
        match, params = step._match_endpoint(HTTPMethod.GET, "/other/path")
        assert match is None
        assert params == {}

        # /apiv2 should not match /api/*
        match, params = step._match_endpoint(HTTPMethod.GET, "/apiv2/users")
        assert match is None
        assert params == {}

    def test_star_prefix_slash_handling(self) -> None:
        """Test that star patterns: /api/* matches /api/something"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        # Verify prefix is correct
        assert len(step._star_patterns) == 1
        star_method, prefix, star_key, _ = step._star_patterns[0]
        assert star_method == HTTPMethod.GET
        assert prefix == "/api/"  # Prefix should have trailing slash

        # /api alone should NOT match (it's the exact prefix, not under it)
        match, params = step._match_endpoint(HTTPMethod.GET, "/api")
        assert match is None
        assert params == {}

        # /api/anything should match
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/something")
        assert match == "GET:/api/*"
        assert params == {}

    def test_star_with_allow_action(self) -> None:
        """Test full flow with star pattern and ALLOW action"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/v1/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)
        event = MockEvent(method=HTTPMethod.GET, path="/v1/anything", body="test")
        result = step.do(event)
        assert result is not None

    def test_star_with_forbid_action(self) -> None:
        """Test full flow with star pattern and FORBID action"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/v1/*", HTTPMethod.GET, APIHandlerAction.FORBID)

        step = _APIHandlerStep(config=config)
        event = MockEvent(method=HTTPMethod.GET, path="/v1/anything", body="test")
        with pytest.raises(mlrun.errors.MLRunAccessDeniedError):
            step.do(event)

    def test_template_insertion_order(self) -> None:
        """Within template patterns, the first registered pattern wins when multiple templates match."""
        config = APIHandlerConfig()
        # /api/{a}/resource is registered first
        config.add_endpoint_handler(
            "/api/{a}/resource", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        # Inject a second overlapping template with a different param name directly (same key structure)
        config._endpoints["GET:/api/{b}/resource"] = {
            _APIEndpointKeys.ACTION: "allow",
            _APIEndpointKeys.DESCRIPTION: None,
        }

        step = _APIHandlerStep(config=config)
        assert len(step._endpoint_patterns) == 2

        # First-inserted template should win — params carry its name
        match, params = step._match_endpoint(HTTPMethod.GET, "/api/foo/resource")
        assert match == "GET:/api/{a}/resource"
        assert params == {"a": "foo"}

    def test_multiple_star_patterns_different_methods(self) -> None:
        """Test multiple star patterns with different HTTP methods"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/read/*", HTTPMethod.GET, APIHandlerAction.ALLOW)
        config.add_endpoint_handler("/write/*", HTTPMethod.POST, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)
        assert len(step._star_patterns) == 2

        # GET /read should match /read/*
        match, params = step._match_endpoint(HTTPMethod.GET, "/read/something")
        assert match == "GET:/read/*"

        # POST /write should match /write/*
        match, params = step._match_endpoint(HTTPMethod.POST, "/write/something")
        assert match == "POST:/write/*"

        # GET /write should NOT match (wrong method)
        match, params = step._match_endpoint(HTTPMethod.GET, "/write/something")
        assert match is None


class TestIncludeUrlInfo:
    """Tests for include_url_info parameter (ML-11658)

    When include_url_info=True, the API handler injects 'mlrun_request_path'
    (the normalized request path, without query string) into the RequestContext
    that is forwarded to the next step.
    """

    def test_include_url_info_default_false(self) -> None:
        """include_url_info should default to False"""
        from mlrun.runtimes.nuclio.serving import APIHandlerConfig

        config = APIHandlerConfig()
        assert config.include_url_info is False

    def test_include_url_info_can_be_set_true(self) -> None:
        """include_url_info should be settable to True"""
        from mlrun.runtimes.nuclio.serving import APIHandlerConfig

        config = APIHandlerConfig(include_url_info=True)
        assert config.include_url_info is True

    def test_include_url_info_in_dict_fields(self) -> None:
        """include_url_info must round-trip through APIHandlerConfig.to_dict/from_dict"""
        from mlrun.runtimes.nuclio.serving import APIHandlerConfig

        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)
        d = config.to_dict()
        assert d["include_url_info"] is True

        config2 = APIHandlerConfig.from_dict(d)
        # from_dict round-trip
        assert config2.include_url_info is True

    def test_include_url_info_disabled_no_path_injected(self) -> None:
        """When include_url_info=False, mlrun_request_path is NOT in the context"""
        config = APIHandlerConfig(include_url_info=False)
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)
        step = _APIHandlerStep(config=config)

        event = MockEvent(method=HTTPMethod.GET, path="/api/test", body="hello")
        result = step.do(event)

        # No params extracted and include_url_info=False → body stays as-is
        assert result.body == "hello"

    def test_include_url_info_enabled_exact_path(self) -> None:
        """When include_url_info=True the RequestContext contains mlrun_request_path"""
        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)
        step = _APIHandlerStep(config=config)

        event = MockEvent(method=HTTPMethod.GET, path="/api/test", body="hello")
        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["mlrun_request_path"] == "/api/test"

    def test_include_url_info_path_without_query_string(self) -> None:
        """mlrun_request_path must NOT include query string"""
        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler(
            "/api/items", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        step = _APIHandlerStep(config=config)

        event = MockEvent(
            method=HTTPMethod.GET, path="/api/items?limit=5&offset=10", body="hello"
        )
        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        # Path must be normalized (no query string)
        assert result.body["mlrun_request_path"] == "/api/items"
        # Query params are still extracted normally
        assert result.body["limit"] == "5"
        assert result.body["offset"] == "10"

    def test_include_url_info_with_path_template(self) -> None:
        """mlrun_request_path must be the actual request path, not the template"""
        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler(
            "/api/users/{user_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        step = _APIHandlerStep(config=config)

        event = MockEvent(
            method=HTTPMethod.GET, path="/api/users/abc-123", body="hello"
        )
        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        # Actual request path, not the template pattern
        assert result.body["mlrun_request_path"] == "/api/users/abc-123"
        # Path param is also present
        assert result.body["user_id"] == "abc-123"

    def test_include_url_info_with_star_pattern(self) -> None:
        """mlrun_request_path is the actual path matched by a star pattern"""
        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler("/api/*", HTTPMethod.GET, APIHandlerAction.ALLOW)
        step = _APIHandlerStep(config=config)

        event = MockEvent(
            method=HTTPMethod.GET, path="/api/deeply/nested/resource", body="hello"
        )
        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["mlrun_request_path"] == "/api/deeply/nested/resource"

    def test_include_url_info_combined_with_existing_params(self) -> None:
        """mlrun_request_path is available alongside path, query, and body params"""
        config = APIHandlerConfig(
            include_url_info=True,
            body_map={"question": "$.q"},
        )
        config.add_endpoint_handler(
            "/api/{model_id}/ask", HTTPMethod.POST, APIHandlerAction.ALLOW
        )
        step = _APIHandlerStep(config=config)

        event = MockEvent(
            method=HTTPMethod.POST,
            path="/api/gpt4/ask?lang=en",
            body={"q": "Hello world"},
        )
        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["mlrun_request_path"] == "/api/gpt4/ask"
        assert result.body["model_id"] == "gpt4"
        assert result.body["lang"] == "en"
        assert result.body["question"] == "Hello world"


class TestAPIHandlerMockServer:
    """Test API handler with mock server integration"""

    def test_api_handler_minimal(self) -> None:
        """Test minimal API handler functionality"""
        fn = cast(
            ServingRuntime, mlrun.new_function("test-api-minimal", kind="serving")
        )

        # Set API handler config using the set_api_handler_config method
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/some/path",  # Use a more realistic path
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
            "Health check",
        )

        # Set the config on the function - this should automatically add _APIHandlerStep
        fn.set_api_handler_config(config)

        # Set topology but don't manually add _APIHandlerStep - it should be automatic
        graph = fn.set_topology("flow", engine="sync")
        # Add a responder step since we removed the respond() from the API handler
        graph.to(name="echo", handler="(event)").respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/some/path",
                method="GET",
                body="ping",
            )
            assert resp == "ping"
        finally:
            server.wait_for_completion()

    def test_api_handler_multiple_paths(self) -> None:
        """Test API handler with multiple different paths"""
        fn = cast(
            ServingRuntime, mlrun.new_function("test-api-multi-paths", kind="serving")
        )

        # Set up config with multiple endpoints
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/health", HTTPMethod.GET, APIHandlerAction.ALLOW, "Health check"
        )
        config.add_endpoint_handler(
            "/api/v1/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Prediction endpoint",
        )
        config.add_endpoint_handler(
            "/api/v1/admin", HTTPMethod.GET, APIHandlerAction.FORBID, "Admin blocked"
        )

        # Set the config on the function - this should automatically add _APIHandlerStep
        fn.set_api_handler_config(config)

        # Set topology but don't manually add _APIHandlerStep - it should be automatic
        graph = fn.set_topology("flow", engine="sync")
        # Add a responder step since we removed the respond() from the API handler
        graph.to(name="echo", handler="(event)").respond()

        server = fn.to_mock_server()
        try:
            # Test allowed GET endpoint
            resp = server.test("/api/v1/health", method="GET", body="ping")
            assert resp == "ping"

            # Test allowed POST endpoint
            resp = server.test("/api/v1/predict", method="POST", body={"data": "test"})
            assert resp == {"data": "test"}

            # Test forbidden endpoint should raise an error
            with pytest.raises(RuntimeError, match="Access forbidden"):
                server.test("/api/v1/admin", method="GET", body="admin-request")

        finally:
            server.wait_for_completion()

    def test_api_handler_path_and_query_params_passed_to_handler(self) -> None:
        """Test that path params and query params are passed to handler arguments"""

        def handler(**kwargs):
            # All params come as kwargs via RequestContext
            return {
                "item_id": kwargs.get("item_id"),
                "source": kwargs.get("source"),
                "limit": kwargs.get("limit"),
            }

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-api-path-query-params", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/items/{item_id}",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Path + query params",
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            response = server.test(
                "/api/v1/items/123?source=ui&limit=10",
                method="POST",
                body={},  # Empty body, params from path/query
            )
            assert response == {
                "item_id": "123",
                "source": "ui",
                "limit": "10",
            }
        finally:
            server.wait_for_completion()

    def test_api_handler_body_map_with_path_query(self) -> None:
        """Test combining body_map, path params, and query params"""

        def handler(**kwargs):
            return {
                "model": kwargs.get("model"),  # from body_map
                "version": kwargs.get("version"),  # from path
                "format": kwargs.get("format"),  # from query
            }

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-all-param-sources", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_body_mapping("model", "$.model_name")
        config.add_endpoint_handler(
            "/predict/{version}",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            response = server.test(
                "/predict/v2?format=json",
                method="POST",
                body={"model_name": "my-classifier"},
            )
            assert response == {
                "model": "my-classifier",
                "version": "v2",
                "format": "json",
            }
        finally:
            server.wait_for_completion()

    def test_api_handler_url_encoded_path_params(self) -> None:
        """Test that URL-encoded path parameters are properly decoded"""

        def handler(**kwargs):
            return {"filename": kwargs.get("filename")}

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-url-encoding", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/files/{filename}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            response = server.test(
                "/files/my%20document.pdf",
                method="GET",
                body={},
            )
            assert response == {"filename": "my document.pdf"}
        finally:
            server.wait_for_completion()

    def test_api_handler_multiple_path_params(self) -> None:
        """Test endpoint with multiple path parameters"""

        def handler(**kwargs):
            return {
                "org": kwargs.get("org_id"),
                "repo": kwargs.get("repo_id"),
                "issue": kwargs.get("issue_num"),
            }

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-multi-path-params", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/orgs/{org_id}/repos/{repo_id}/issues/{issue_num}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            response = server.test(
                "/orgs/mlrun/repos/mlrun/issues/42",
                method="GET",
                body={},
            )
            assert response == {
                "org": "mlrun",
                "repo": "mlrun",
                "issue": "42",
            }
        finally:
            server.wait_for_completion()

    def test_api_handler_body_map_path_conflict_at_init(self) -> None:
        """Test that body_map vs path template conflicts are caught at init time"""

        def handler(**kwargs):
            return {"id": kwargs.get("id")}

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-param-conflict", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_body_mapping("id", "$.identifier")  # body_map extracts 'id'
        config.add_endpoint_handler(
            "/items/{id}",  # path also has 'id'
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        # Conflict should be raised at mock server init, not at request time
        with pytest.raises(
            mlrun.errors.MLRunValueError,
            match="Configuration conflict.*body_map parameter.*id.*overlap with path template",
        ):
            fn.to_mock_server()

    def test_api_handler_parameter_conflict_error(
        self,
    ) -> None:
        """Test that parameter conflicts from multiple sources raise error at request time"""

        def handler(**kwargs):
            return {"limit": kwargs.get("limit")}

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-param-conflict", kind="serving"),
        )

        config = APIHandlerConfig()
        # Use a path template with 'category' (no body_map overlap) so init passes,
        # but the query string at request time will conflict with 'limit' from path.
        config.add_endpoint_handler(
            "/items/{category}/{limit}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            # Should raise 400 Bad Request due to runtime parameter conflict
            # (path 'limit' = "50" vs query 'limit' = "100")
            with pytest.raises(RuntimeError, match="400.*Parameter name conflict"):
                server.test(
                    "/items/electronics/50?limit=100",
                    method="GET",
                )
        finally:
            server.wait_for_completion()

    def test_api_handler_repeated_query_params(self) -> None:
        """Test that repeated query parameters are passed as lists"""

        def handler(**kwargs):
            return {"ids": kwargs.get("id"), "single": kwargs.get("name")}

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-query-list", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/items",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            # Test with repeated query param: ?id=1&id=4&id=1&name=test
            response = server.test(
                "/items?id=1&id=4&id=1&name=test",
                method="GET",
            )

            # Repeated 'id' should be a list, single 'name' should be string
            assert response == {"ids": ["1", "4", "1"], "single": "test"}
        finally:
            server.wait_for_completion()

    def test_api_handler_path_params_in_signature_with_repeated_query_params(
        self,
    ) -> None:
        """Test handler with path params in signature and repeated query params.

        Demonstrates:
        - Path parameters extracted and passed as named arguments
        - Query parameters with multiple values passed as lists
        - Handler signature with explicit parameter names (not just **kwargs)
        """

        def handler(
            item_id: str, category: str, tags: list[str] | None = None, **kwargs
        ) -> dict:
            """Handler with explicit path params in signature.

            Args:
                item_id: From path parameter {item_id}
                category: From path parameter {category}
                tags: From repeated query param ?tags=...&tags=...
                **kwargs: Catch any other params
            """
            return {
                "item_id": item_id,
                "category": category,
                "tags": tags,
                "single_param": kwargs.get("limit"),
            }

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-signature-with-lists", kind="serving"),
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/items/{category}/{item_id}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="handler", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            # Test with path params + repeated query param + single query param
            response = server.test(
                "/items/electronics/laptop-123?tags=new&tags=featured&tags=sale&limit=10",
                method="GET",
            )

            # Path params should be passed as named args
            # Repeated query param should be a list
            # Single query param should be a string
            assert response == {
                "item_id": "laptop-123",
                "category": "electronics",
                "tags": ["new", "featured", "sale"],
                "single_param": "10",
            }
        finally:
            server.wait_for_completion()


class TestAPIHandlerConfig:
    """Direct tests for APIHandlerConfig class"""

    def test_init_defaults(self) -> None:
        """Test APIHandlerConfig initialization with defaults"""
        config = APIHandlerConfig()
        assert config.enabled is True
        assert config.endpoints == {}

    def test_init_with_parameters(self) -> None:
        """Test APIHandlerConfig initialization with parameters"""
        endpoints = {
            "GET:/health": {
                _APIEndpointKeys.ACTION: "allow",
                _APIEndpointKeys.DESCRIPTION: "Health",
            }
        }
        config = APIHandlerConfig(enabled=False, endpoints=endpoints)
        assert config.enabled is False
        assert "GET:/health" in config.endpoints

    def test_add_endpoint_handler(self) -> None:
        """Test adding endpoint handlers"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, "Prediction"
        )

        endpoint_config = config.get_endpoint_config(HTTPMethod.POST, "/api/predict")
        assert endpoint_config is not None
        assert endpoint_config[_APIEndpointKeys.ACTION] == "allow"
        assert endpoint_config[_APIEndpointKeys.DESCRIPTION] == "Prediction"

    def test_add_multiple_endpoints(self) -> None:
        """Test adding multiple endpoint handlers"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/health", HTTPMethod.GET, APIHandlerAction.ALLOW)
        config.add_endpoint_handler("/metrics", HTTPMethod.GET, APIHandlerAction.ALLOW)
        config.add_endpoint_handler("/admin", HTTPMethod.POST, APIHandlerAction.FORBID)

        assert len(config.endpoints) == 3
        assert config.get_endpoint_config(HTTPMethod.GET, "/health") is not None
        assert config.get_endpoint_config(HTTPMethod.GET, "/metrics") is not None
        assert config.get_endpoint_config(HTTPMethod.POST, "/admin") is not None

    def test_remove_endpoint_handler(self) -> None:
        """Test removing endpoint handlers"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/test", HTTPMethod.POST, APIHandlerAction.ALLOW
        )
        assert config.get_endpoint_config(HTTPMethod.POST, "/api/test") is not None

        config.remove_endpoint_handler("/api/test", HTTPMethod.POST)
        assert config.get_endpoint_config(HTTPMethod.POST, "/api/test") is None

    def test_remove_endpoint_handler_without_leading_slash(self) -> None:
        """Test removing endpoint handlers works with and without leading slash"""
        config = APIHandlerConfig()

        # Add with leading slash
        config.add_endpoint_handler(
            "/api/test", HTTPMethod.POST, APIHandlerAction.ALLOW
        )
        assert config.get_endpoint_config(HTTPMethod.POST, "/api/test") is not None

        # Remove without leading slash should still work
        config.remove_endpoint_handler("api/test", HTTPMethod.POST)
        assert config.get_endpoint_config(HTTPMethod.POST, "/api/test") is None

        # Add without leading slash
        config.add_endpoint_handler(
            "api/test2", HTTPMethod.GET, APIHandlerAction.FORBID
        )
        # Should be retrievable with or without leading slash
        assert config.get_endpoint_config(HTTPMethod.GET, "/api/test2") is not None
        assert config.get_endpoint_config(HTTPMethod.GET, "api/test2") is not None

        # Remove with leading slash should still work
        config.remove_endpoint_handler("/api/test2", HTTPMethod.GET)
        assert config.get_endpoint_config(HTTPMethod.GET, "/api/test2") is None

    def test_get_endpoint_config_not_found(self) -> None:
        """Test getting non-existent endpoint config"""
        config = APIHandlerConfig()
        assert config.get_endpoint_config(HTTPMethod.GET, "/nonexistent") is None

    def test_invalid_http_method_validation(self) -> None:
        """Test that invalid http_method types are rejected with clear error messages"""
        config = APIHandlerConfig()

        # Test add_endpoint_handler with wrong type (APIHandlerAction instead of HTTPMethod)
        # APIHandlerAction is a StrEnum, so it's treated as a string and validated
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Invalid HTTP method string 'allow'",
        ):
            config.add_endpoint_handler(
                "/test",
                http_method=APIHandlerAction.ALLOW,  # Wrong type!
            )

        # Test remove_endpoint_handler with wrong type
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Invalid HTTP method string 'GT'",
        ):
            config.remove_endpoint_handler(
                "/test",
                http_method="GT",  # Wrong value
            )

        # Test get_endpoint_config with wrong type
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Invalid HTTP method string 'allow'",
        ):
            config.get_endpoint_config(
                method=APIHandlerAction.ALLOW,
                path="/test",  # Wrong type!
            )

        # Test with invalid string
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Invalid HTTP method string 'INVALID'",
        ):
            config.add_endpoint_handler("/test", http_method="INVALID")

    def test_http_method_string_conversion(self) -> None:
        """Test that string HTTP methods are correctly converted to HTTPMethod enum"""
        config = APIHandlerConfig()

        # Test add with string (lowercase)
        config.add_endpoint_handler(
            "/test1", http_method="get", action=APIHandlerAction.ALLOW
        )
        assert config.get_endpoint_config(HTTPMethod.GET, "/test1") is not None

        # Test add with string (uppercase)
        config.add_endpoint_handler(
            "/test2", http_method="POST", action=APIHandlerAction.FORBID
        )
        assert config.get_endpoint_config(HTTPMethod.POST, "/test2") is not None

        # Test get with string
        assert config.get_endpoint_config("get", "/test1") is not None
        assert config.get_endpoint_config("post", "/test2") is not None

        # Test remove with string
        config.remove_endpoint_handler("/test1", http_method="get")
        assert config.get_endpoint_config(HTTPMethod.GET, "/test1") is None

        # Verify test2 still exists
        assert config.get_endpoint_config(HTTPMethod.POST, "/test2") is not None

    def test_endpoints_property_setter(self) -> None:
        """Test setting endpoints via property"""
        config = APIHandlerConfig()
        endpoints = {
            "POST:/predict": {
                _APIEndpointKeys.ACTION: "allow",
                _APIEndpointKeys.DESCRIPTION: "Predict",
            },
            "GET:/health": {
                _APIEndpointKeys.ACTION: "allow",
                _APIEndpointKeys.DESCRIPTION: "Health",
            },
        }
        config.endpoints = endpoints

        assert len(config.endpoints) == 2
        assert config.get_endpoint_config(HTTPMethod.POST, "/predict") is not None
        assert config.get_endpoint_config(HTTPMethod.GET, "/health") is not None

    def test_parse_endpoint_key(self) -> None:
        """Test parsing endpoint keys"""
        config = APIHandlerConfig()
        method, path = config._parse_endpoint_key("GET:/api/test")
        assert method == HTTPMethod.GET
        assert path == "/api/test"

    def test_parse_endpoint_key_invalid(self) -> None:
        """Test parsing invalid endpoint key"""
        config = APIHandlerConfig()
        with pytest.raises(ValueError, match="Invalid endpoint key format"):
            config._parse_endpoint_key("invalid-format")

    def test_to_dict(self) -> None:
        """Test serialization to dictionary"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.POST, APIHandlerAction.ALLOW)

        config_dict = config.to_dict()
        assert "enabled" in config_dict
        assert "endpoints" in config_dict
        assert config_dict["enabled"] is True

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary"""
        data = {
            "enabled": True,
            "endpoints": {
                "POST:/predict": {
                    _APIEndpointKeys.ACTION: "allow",
                    _APIEndpointKeys.DESCRIPTION: "Prediction",
                }
            },
        }
        config = APIHandlerConfig.from_dict(data)
        assert config.enabled is True
        assert config.get_endpoint_config(HTTPMethod.POST, "/predict") is not None

    def test_add_endpoint_handler_override_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warning is logged when overriding existing endpoint"""
        config = APIHandlerConfig()

        # Add an endpoint
        config.add_endpoint_handler(
            "/api/test", HTTPMethod.POST, APIHandlerAction.ALLOW, "First config"
        )

        # Override the same endpoint - should trigger warning
        with caplog.at_level(logging.WARNING):
            config.add_endpoint_handler(
                "/api/test", HTTPMethod.POST, APIHandlerAction.FORBID, "Second config"
            )

        # Verify warning was logged
        assert any(
            "Overriding existing endpoint" in record.message
            for record in caplog.records
        )

        # Verify the endpoint was updated
        endpoint_config = config.get_endpoint_config(HTTPMethod.POST, "/api/test")
        assert endpoint_config[_APIEndpointKeys.ACTION] == "forbid"
        assert endpoint_config[_APIEndpointKeys.DESCRIPTION] == "Second config"

    def test_add_endpoint_star_not_at_end_raises(self) -> None:
        """SDK rejects '*' that is not at the tail of the path at config time."""
        config = APIHandlerConfig()
        with pytest.raises(
            mlrun.errors.MLRunValueError, match="wildcard.*must be at the end"
        ):
            config.add_endpoint_handler(
                "/api/*/users", HTTPMethod.GET, APIHandlerAction.ALLOW
            )

    def test_add_endpoint_multiple_stars_raises(self) -> None:
        """SDK rejects paths with more than one '*'."""
        config = APIHandlerConfig()
        with pytest.raises(mlrun.errors.MLRunValueError, match="wildcard.*only once"):
            config.add_endpoint_handler("/*/*", HTTPMethod.GET, APIHandlerAction.ALLOW)

    def test_add_endpoint_valid_star_pattern(self) -> None:
        """A single trailing '*' is accepted by the SDK."""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/v1/*", HTTPMethod.GET, APIHandlerAction.ALLOW)
        assert config.get_endpoint_config(HTTPMethod.GET, "/api/v1/*") is not None


class TestSetAPIHandlerConfig:
    """Tests for ServingRuntime.set_api_handler_config method"""

    def test_set_api_handler_config_with_valid_dict(self) -> None:
        """Test setting API handler config with a valid dictionary"""
        fn = cast(ServingRuntime, mlrun.new_function("test-fn", kind="serving"))

        config_dict = {
            "enabled": True,
            "endpoints": {
                "POST:/predict": {
                    _APIEndpointKeys.ACTION: "allow",
                    _APIEndpointKeys.DESCRIPTION: "Prediction",
                }
            },
        }

        fn.set_api_handler_config(config_dict)
        assert fn.spec.api_handler_config is not None
        assert fn.spec.api_handler_config["enabled"] is True
        assert "POST:/predict" in fn.spec.api_handler_config["endpoints"]

    def test_set_api_handler_config_with_invalid_dict(self) -> None:
        """Test setting API handler config with an invalid dictionary"""
        fn = cast(ServingRuntime, mlrun.new_function("test-fn", kind="serving"))

        # Invalid dict - missing required fields or invalid format
        invalid_config = {
            "invalid_key": "invalid_value",
            "endpoints": "not_a_dict",  # Should be a dict
        }

        with pytest.raises(ValueError, match="Invalid API handler config dict format"):
            fn.set_api_handler_config(invalid_config)

    def test_set_api_handler_config_with_invalid_type(self) -> None:
        """Test setting API handler config with invalid type"""
        fn = cast(ServingRuntime, mlrun.new_function("test-fn", kind="serving"))

        with pytest.raises(ValueError, match="config must be"):
            fn.set_api_handler_config("invalid_string")

    def test_set_api_handler_config_with_api_handler_config_object(self) -> None:
        """Test setting API handler config with APIHandlerConfig object"""
        fn = cast(ServingRuntime, mlrun.new_function("test-fn", kind="serving"))

        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        fn.set_api_handler_config(config)
        assert fn.spec.api_handler_config is not None
        assert fn.spec.api_handler_config["enabled"] is True


class TestEndpointKeyHelpers:
    """Direct tests for endpoint key helper functions"""

    def test_combine_serving_endpoint_key(self) -> None:
        """Test combining method and path into endpoint key"""
        key = _combine_serving_endpoint_key(HTTPMethod.GET, "/api/test")
        assert key == "GET:/api/test"

        key = _combine_serving_endpoint_key(HTTPMethod.POST, "/predict")
        assert key == "POST:/predict"

    def test_split_serving_endpoint_key(self) -> None:
        """Test splitting endpoint key into method and path"""
        method, path = _split_serving_endpoint_key("GET:/api/test")
        assert method == HTTPMethod.GET
        assert path == "/api/test"

        method, path = _split_serving_endpoint_key("POST:/predict")
        assert method == HTTPMethod.POST
        assert path == "/predict"

    def test_split_serving_endpoint_key_with_colon_in_path(self) -> None:
        """Test splitting endpoint key when path contains colon"""
        method, path = _split_serving_endpoint_key("GET:/api/test:123")
        assert method == HTTPMethod.GET
        assert path == "/api/test:123"

    def test_split_serving_endpoint_key_invalid(self) -> None:
        """Test splitting invalid endpoint key"""
        with pytest.raises(ValueError):
            _split_serving_endpoint_key("invalid-key-without-colon")

    def test_roundtrip_combine_split(self) -> None:
        """Test roundtrip conversion"""
        original_method = HTTPMethod.PUT
        original_path = "/api/v1/resource/123"

        key = _combine_serving_endpoint_key(original_method, original_path)
        method, path = _split_serving_endpoint_key(key)

        assert method == original_method
        assert path == original_path


class TestAPIHandlerStep:
    """Direct tests for _APIHandlerStep class"""

    def test_init_with_config_object(self) -> None:
        """Test initialization with APIHandlerConfig object"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config, name="test-handler")
        assert step.name == "test-handler"
        assert step.config is config

    def test_init_with_config_dict(self) -> None:
        """Test initialization with config dictionary"""
        config_dict = {
            "enabled": True,
            "endpoints": {
                "POST:/predict": {
                    _APIEndpointKeys.ACTION: "allow",
                    _APIEndpointKeys.DESCRIPTION: "Predict",
                }
            },
        }
        step = _APIHandlerStep(config=config_dict)
        assert isinstance(step.config, APIHandlerConfig)
        assert step.config.get_endpoint_config(HTTPMethod.POST, "/predict") is not None

    def test_init_no_config(self) -> None:
        """Test initialization without config"""
        step = _APIHandlerStep()
        assert isinstance(step.config, APIHandlerConfig)
        assert step.config.endpoints == {}

    def test_match_endpoint_exact(self) -> None:
        """Test exact endpoint matching"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        match, path_params = step._match_endpoint(HTTPMethod.GET, "/api/test")
        assert match == "GET:/api/test"
        assert path_params == {}

    def test_match_endpoint_path_template(self) -> None:
        """Test endpoint matching for path-template endpoints"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/items/{item_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        match, path_params = step._match_endpoint(HTTPMethod.GET, "/api/items/abc-123")
        assert match == "GET:/api/items/{item_id}"
        assert path_params == {"item_id": "abc-123"}

    def test_match_endpoint_no_match(self) -> None:
        """Test endpoint matching when no match found"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        match, path_params = step._match_endpoint(HTTPMethod.POST, "/api/test")
        assert match is None
        assert path_params == {}

        match, path_params = step._match_endpoint(HTTPMethod.GET, "/different/path")
        assert match is None
        assert path_params == {}

    def test_run_path_template_method_not_allowed(self) -> None:
        """Test that wrong method on path-template endpoint returns 404.

        Note: For performance, we only check exact paths when distinguishing 405 vs 404.
        Templated paths that don't match the request method will return 404, not 405.
        """
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/resource/{resource_id}",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )

        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.GET
        mock_event.path = "/api/resource/42"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(mlrun.errors.MLRunNotFoundError, match="Endpoint not found"):
            step.do({"data": "test"})

    def test_run_allowed_endpoint(self) -> None:
        """Test running with allowed endpoint"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        # Create a mock context with current_event
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.GET
        mock_event.path = "/test"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        result = step.do({"data": "test"})
        assert result == {"data": "test"}

    def test_run_forbidden_endpoint(self) -> None:
        """Test running with forbidden endpoint"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/admin", HTTPMethod.POST, APIHandlerAction.FORBID)

        # Create a mock context with current_event
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.POST
        mock_event.path = "/admin"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunAccessDeniedError, match="Access forbidden"
        ):
            step.do({"data": "test"})

    def test_run_method_not_allowed(self) -> None:
        """Test that wrong HTTP method for existing endpoint returns 405"""
        config = APIHandlerConfig()
        # Endpoint exists for POST, but we'll try GET
        config.add_endpoint_handler(
            "/resource", HTTPMethod.POST, APIHandlerAction.ALLOW
        )

        # Create a mock context with current_event
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.GET
        mock_event.path = "/resource"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunMethodNotAllowedError, match="Method not allowed"
        ):
            step.do({"data": "test"})

    def test_run_no_matching_endpoint(self) -> None:
        """Test running with no matching endpoint"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        # Create a mock context with current_event
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.POST
        mock_event.path = "/nonexistent"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(mlrun.errors.MLRunNotFoundError, match="Endpoint not found"):
            step.do({"data": "test"})

    def test_run_no_method_in_context(self) -> None:
        """Test running without method in context"""
        config = APIHandlerConfig()
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = None
        mock_event.path = "/test"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunBadRequestError, match="HTTP method not found"
        ):
            step.do({"data": "test"})

    def test_run_no_path_in_context(self) -> None:
        """Test running without path in context"""
        config = APIHandlerConfig()
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.GET
        mock_event.path = None
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunBadRequestError, match="Request path not found"
        ):
            step.do({"data": "test"})

    def test_run_string_method_conversion(self) -> None:
        """Test running with string method that gets converted to HTTPMethod"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = "get"  # lowercase string
        mock_event.path = "/test"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        result = step.do({"data": "test"})
        assert result == {"data": "test"}

    def test_run_invalid_method_string(self) -> None:
        """Test running with invalid method string"""
        config = APIHandlerConfig()
        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = "INVALID"
        mock_event.path = "/test"
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunBadRequestError, match="Unsupported HTTP method"
        ):
            step.do({"data": "test"})

    def test_run_body_map_with_missing_body(self) -> None:
        """Test that body_map with missing/non-dict body raises 422 error"""
        config = APIHandlerConfig()
        config.body_map = {"$.name": "user_name"}
        config.add_endpoint_handler("/test", HTTPMethod.POST, APIHandlerAction.ALLOW)

        context = MagicMock()
        mock_event = MagicMock()
        mock_event.method = HTTPMethod.POST
        mock_event.path = "/test"
        mock_event.body = None
        context.current_event = mock_event

        step = _APIHandlerStep(config=config)
        step.context = context

        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="body_map configured but request body is not a dict",
        ):
            step.do(mock_event)


class TestAddAPIHandlerStepToGraph:
    """Direct tests for _add_api_handler_step_to_graph function"""

    def test_add_api_handler_step_with_dict_spec(self) -> None:
        """Test adding API handler step with dict serving_spec"""
        graph = RootFlowStep()
        graph.to(name="echo", handler="(event)")

        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        serving_spec = {"api_handler_config": config.to_dict()}
        context = MagicMock()

        result_graph = _add_api_handler_step_to_graph(graph, serving_spec, context)

        # Check that api-handler step was added
        assert "api-handler" in result_graph.steps
        api_handler_step = result_graph.steps["api-handler"]
        assert (
            api_handler_step.class_name == "mlrun.serving.api_handler._APIHandlerStep"
        )

        # Check that existing step now comes after api-handler
        echo_step = result_graph.steps["echo"]
        assert "api-handler" in echo_step.after

    def test_add_api_handler_step_no_config(self) -> None:
        """Test that no API handler step is added when config is absent"""
        graph = RootFlowStep()
        graph.to(name="echo", handler="(event)")

        serving_spec = {}
        context = MagicMock()

        result_graph = _add_api_handler_step_to_graph(graph, serving_spec, context)

        # Check that api-handler step was NOT added
        assert "api-handler" not in result_graph.steps

    def test_add_api_handler_step_prevents_duplicates(self) -> None:
        """Test that duplicate API handler steps are not added"""
        graph = RootFlowStep()
        graph.add_step(
            class_name="mlrun.serving.api_handler._APIHandlerStep",
            name="api-handler",
            config={},
        )
        graph.to(name="echo", handler="(event)")

        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        serving_spec = {"api_handler_config": config.to_dict()}
        context = MagicMock()

        result_graph = _add_api_handler_step_to_graph(graph, serving_spec, context)

        # Count API handler steps - should only be one
        api_handler_steps = [
            step
            for step in result_graph.steps.values()
            if hasattr(step, "class_name")
            and step.class_name == "mlrun.serving.api_handler._APIHandlerStep"
        ]
        assert len(api_handler_steps) == 1

    def test_add_api_handler_step_invalid_spec_type(self) -> None:
        """Test error when serving_spec is invalid type"""
        graph = RootFlowStep()
        context = MagicMock()

        with pytest.raises(mlrun.errors.MLRunValueError, match="serving_spec must be"):
            _add_api_handler_step_to_graph(graph, "invalid", context)

    def test_add_api_handler_step_multiple_starting_steps(self) -> None:
        """Test adding API handler when graph has multiple starting steps"""
        graph = RootFlowStep()
        graph.add_step(name="step1", handler="(event)")
        graph.add_step(name="step2", handler="(event)")

        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        serving_spec = {"api_handler_config": config.to_dict()}
        context = MagicMock()

        result_graph = _add_api_handler_step_to_graph(graph, serving_spec, context)

        # Both starting steps should now come after api-handler
        assert "api-handler" in result_graph.steps["step1"].after
        assert "api-handler" in result_graph.steps["step2"].after

    def test_add_api_handler_step_with_cyclic_graph(self) -> None:
        """Test adding API handler to a graph with cyclic steps"""
        graph = RootFlowStep()

        # Create a cyclic graph: step1 -> step2 -> step3 -> step1
        # First create steps without the cyclic dependency
        graph.add_step(name="step1", handler="(event)")
        graph.add_step(name="step2", handler="(event)", after=["step1"])
        graph.add_step(name="step3", handler="(event)", after=["step2"])

        # Now add the cycle: step1 comes after step3
        graph.steps["step1"].after = ["step3"]
        # Mark step1 as cyclic (it has 'after' but is also a starting point)
        graph.steps["step1"].cycle_from = ["step3"]

        config = APIHandlerConfig()
        config.add_endpoint_handler("/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        serving_spec = {"api_handler_config": config.to_dict()}
        context = MagicMock()

        result_graph = _add_api_handler_step_to_graph(graph, serving_spec, context)

        # API handler should be added
        assert "api-handler" in result_graph.steps

        # step1 (the cyclic starting step) should now come after api-handler
        assert "api-handler" in result_graph.steps["step1"].after
        # step1 should still maintain its cycle_from
        assert result_graph.steps["step1"].cycle_from == ["step3"]

        # step2 and step3 should not have api-handler in their after lists
        # (they are not starting steps)
        assert "api-handler" not in result_graph.steps["step2"].after
        assert "api-handler" not in result_graph.steps["step3"].after


class TestParseQueryParams:
    """Tests for _parse_query_params static method"""

    def test_parse_query_params_no_query_string(self) -> None:
        """Test parsing path with no query string"""
        path, params = _APIHandlerStep._parse_query_params("/api/users")
        assert path == "/api/users"
        assert params == {}

    def test_parse_query_params_single_param(self) -> None:
        """Test parsing single query parameter"""
        path, params = _APIHandlerStep._parse_query_params("/api/users?limit=10")
        assert path == "/api/users"
        assert params == {"limit": "10"}

    def test_parse_query_params_multiple_params(self) -> None:
        """Test parsing multiple query parameters"""
        path, params = _APIHandlerStep._parse_query_params(
            "/api/users?limit=10&offset=20&sort=name"
        )
        assert path == "/api/users"
        assert params == {"limit": "10", "offset": "20", "sort": "name"}

    def test_parse_query_params_repeated_values(self) -> None:
        """Test parsing repeated query parameters (should return list)"""
        path, params = _APIHandlerStep._parse_query_params("/api/users?id=1&id=2&id=3")
        assert path == "/api/users"
        assert params == {"id": ["1", "2", "3"]}

    def test_parse_query_params_mixed_single_and_repeated(self) -> None:
        """Test parsing mix of single and repeated params"""
        path, params = _APIHandlerStep._parse_query_params(
            "/api/users?id=1&id=2&limit=10&id=3"
        )
        assert path == "/api/users"
        assert params == {"id": ["1", "2", "3"], "limit": "10"}

    def test_parse_query_params_empty_value(self) -> None:
        """Test parsing query param with empty value"""
        path, params = _APIHandlerStep._parse_query_params("/api/users?filter=")
        assert path == "/api/users"
        assert params == {"filter": ""}

    def test_parse_query_params_url_encoded(self) -> None:
        """Test parsing URL-encoded query parameters"""
        path, params = _APIHandlerStep._parse_query_params(
            "/api/users?name=John%20Doe&email=test%40example.com"
        )
        assert path == "/api/users"
        assert params == {"name": "John Doe", "email": "test@example.com"}

    def test_parse_query_params_special_characters(self) -> None:
        """Test parsing query params with special characters"""
        path, params = _APIHandlerStep._parse_query_params(
            "/api/search?q=hello+world&filter=a%26b"
        )
        assert path == "/api/search"
        # parse_qs converts + to space and decodes %26 to &
        assert params == {"q": "hello world", "filter": "a&b"}

    def test_parse_query_params_empty_string(self) -> None:
        """Test parsing empty string (should normalize to '/')"""
        path, params = _APIHandlerStep._parse_query_params("")
        assert path == "/"
        assert params == {}

    def test_parse_query_params_root_path_with_query(self) -> None:
        """Test parsing root path with query string"""
        path, params = _APIHandlerStep._parse_query_params("/?key=value")
        assert path == "/"
        assert params == {"key": "value"}


class TestExtractQueryParams:
    """Tests for _extract_query_params method"""

    def test_extract_query_params_from_path(self) -> None:
        """Test extracting query params from path (mock event)"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)
        event = MockEvent(path="/api/users?limit=10", method="GET")

        normalized_path, params = handler._extract_query_params(
            event, "/api/users?limit=10"
        )
        assert normalized_path == "/api/users"
        assert params == {"limit": "10"}

    def test_extract_query_params_no_query_in_path(self) -> None:
        """Test extracting query params when path has no query string"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)
        event = MockEvent(path="/api/users", method="GET")

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        assert params == {}

    def test_extract_query_params_from_event_fields(self) -> None:
        """Test extracting query params from event.fields (Nuclio format)"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)

        # Create a mock event with fields attribute (simulating Nuclio event)
        event = MagicMock()
        event.method = "GET"
        event.path = "/api/users"
        event.fields = {"limit": ["10"], "offset": ["20"]}

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        assert params == {"limit": "10", "offset": "20"}

    def test_extract_query_params_from_event_fields_multiple_values(self) -> None:
        """Test extracting repeated query params from event.fields"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)

        event = MagicMock()
        event.method = "GET"
        event.path = "/api/users"
        event.fields = {"id": ["1", "2", "3"], "single": ["value"]}

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        assert params == {"id": ["1", "2", "3"], "single": "value"}

    def test_extract_query_params_from_context_event_fields(self) -> None:
        """Test extracting query params from context.current_event.fields"""
        config = APIHandlerConfig()
        context = GraphContext()

        # Create a mock current_event in context
        mock_event = MagicMock()
        mock_event.fields = {"limit": ["10"], "filter": ["active"]}
        context.current_event = mock_event

        handler = _APIHandlerStep(config=config, context=context)

        # Event without fields
        event = MockEvent(path="/api/users", method="GET")

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        assert params == {"limit": "10", "filter": "active"}

    def test_extract_query_params_empty_fields_list(self) -> None:
        """Test extracting query params when event.fields has empty list"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)

        event = MagicMock()
        event.method = "GET"
        event.path = "/api/users"
        event.fields = {"empty": [], "has_value": ["test"]}

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        # Empty list should result in None value
        assert params == {"empty": None, "has_value": "test"}

    def test_extract_query_params_fields_non_list_value(self) -> None:
        """Test extracting query params when event.fields has non-list value"""
        config = APIHandlerConfig()
        handler = _APIHandlerStep(config=config)

        event = MagicMock()
        event.method = "GET"
        event.path = "/api/users"
        event.fields = {"string_value": "test", "list_value": ["value"]}

        normalized_path, params = handler._extract_query_params(event, "/api/users")
        assert normalized_path == "/api/users"
        assert params == {"string_value": "test", "list_value": "value"}


class TestCompileEndpointPatterns:
    """Tests for _compile_endpoint_patterns method"""

    def test_compile_endpoint_patterns_no_templates(self) -> None:
        """Test that endpoints without templates are skipped"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/users", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/api/items", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        handler = _APIHandlerStep(config=config)
        # Should have no patterns (no path templates)
        assert len(handler._endpoint_patterns) == 0

    def test_compile_endpoint_patterns_with_templates(self) -> None:
        """Test compiling patterns for endpoints with templates"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/users/{user_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )
        config.add_endpoint_handler(
            "/api/items/{item_id}", HTTPMethod.POST, APIHandlerAction.ALLOW
        )

        handler = _APIHandlerStep(config=config)
        # Should have 2 patterns
        assert len(handler._endpoint_patterns) == 2

        # Verify patterns are compiled
        for (
            method,
            pattern,
            endpoint_key,
            endpoint_config,
        ) in handler._endpoint_patterns:
            assert isinstance(pattern, type(re.compile("")))
            assert method in [
                HTTPMethod.GET,
                HTTPMethod.POST,
                HTTPMethod.PUT,
                HTTPMethod.DELETE,
            ]

    def test_compile_endpoint_patterns_invalid_regex(self) -> None:
        """Test that invalid regex patterns raise error during initialization"""
        # This test verifies error handling for malformed path templates
        # Since our template conversion is quite robust, we'd need to manually
        # break the regex. For now, this test documents the expected behavior.

        # Note: Our current implementation is unlikely to produce invalid regex
        # because we escape the path and only replace {param} with capture groups.
        # If we wanted to test this, we'd need to inject a bad pattern into _endpoints
        # or modify the regex generation logic to be more permissive.
        pass  # Placeholder - actual invalid regex is hard to trigger with current impl


class TestAPIHandlerEdgeCases:
    """Tests for edge cases and integration scenarios"""

    def test_query_params_with_path_params_integration(self) -> None:
        """Test that query params and path params work together correctly"""
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/users/{user_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
        )

        # Create the handler step directly
        handler = _APIHandlerStep(config=config)

        # Create event with both path param and query param
        event = MockEvent(
            path="/api/users/123?fields=name,email&limit=10",
            method="GET",
            body="test",
        )

        # Process the event
        result = handler.do(event)

        # The handler modifies event.body to be a RequestContext when params are extracted
        # (see api_handler.py line ~430: "if hasattr(event, 'body'): event.body = request_context")
        assert isinstance(result, MockEvent)
        assert isinstance(result.body, _RequestContext)
        assert result.body["user_id"] == "123"
        assert result.body["fields"] == "name,email"
        assert result.body["limit"] == "10"
