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
from http import HTTPMethod
from typing import cast
from unittest.mock import MagicMock

import pytest

import mlrun
import mlrun.errors
from mlrun.common.schemas.serving import APIHandlerAction, _APIEndpointKeys
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, ServingRuntime
from mlrun.serving import GraphContext
from mlrun.serving.api_handler import _APIHandlerStep
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

    def test_get_endpoint_config_not_found(self) -> None:
        """Test getting non-existent endpoint config"""
        config = APIHandlerConfig()
        assert config.get_endpoint_config(HTTPMethod.GET, "/nonexistent") is None

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

        match = step._match_endpoint(HTTPMethod.GET, "/api/test")
        assert match == "GET:/api/test"

    def test_match_endpoint_no_match(self) -> None:
        """Test endpoint matching when no match found"""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/api/test", HTTPMethod.GET, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        match = step._match_endpoint(HTTPMethod.POST, "/api/test")
        assert match is None

        match = step._match_endpoint(HTTPMethod.GET, "/different/path")
        assert match is None

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

        with pytest.raises(mlrun.errors.MLRunBadRequestError, match="Access forbidden"):
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
