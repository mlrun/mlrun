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

"""Unit tests for JSONPath body_map support in the serving endpoint"""

import logging
from http import HTTPMethod
from typing import cast
from unittest.mock import MagicMock

import pytest

import mlrun
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, ServingRuntime
from mlrun.serving.api_handler import _APIHandlerStep


# ---------------------------------------------------------------------------
# Helper classes for tests
# ---------------------------------------------------------------------------
class PrefixStep:
    """Processing step that adds prefix to mapped values"""

    def __init__(self, prefix: str = "", **kwargs):
        self.prefix = prefix

    def do(self, event):
        # Extract the mapped values and add prefix
        result = f"{self.prefix}: arg1={event['arg1']}, arg2={event['arg2']}"
        return result


# ---------------------------------------------------------------------------
# APIHandlerConfig body_map tests (config-level)
# ---------------------------------------------------------------------------
class TestAPIHandlerConfigBodyMap:
    """Tests for body_map on the APIHandlerConfig level"""

    def test_config_with_body_map(self) -> None:
        """Test creating APIHandlerConfig with a body_map"""
        body_map = {"model_name": "$.model", "input_data": "$.data.inputs"}
        config = APIHandlerConfig(body_map=body_map)
        assert config.body_map == body_map

    def test_config_without_body_map(self) -> None:
        """Test that body_map defaults to empty dict"""
        config = APIHandlerConfig()
        assert config.body_map == {}

    def test_body_map_serialization_roundtrip(self) -> None:
        """Test body_map survives to_dict / from_dict round-trip"""
        body_map = {"user_name": "$.name", "user_email": "$.contact.email"}
        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler(
            "/users", HTTPMethod.POST, APIHandlerAction.ALLOW, "Create user"
        )

        # Serialize
        config_dict = config.to_dict()
        assert config_dict["body_map"] == body_map

        # Deserialize
        restored = APIHandlerConfig.from_dict(config_dict)
        assert restored.body_map == body_map

    def test_body_map_shared_across_endpoints(self) -> None:
        """Test that body_map applies to all endpoints (not per-endpoint)"""
        body_map = {"input": "$.data"}
        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)
        config.add_endpoint_handler(
            "/classify", HTTPMethod.POST, APIHandlerAction.ALLOW
        )

        # body_map is on the config, not on individual endpoints
        assert config.body_map == body_map
        predict_cfg = config.get_endpoint_config(HTTPMethod.POST, "/predict")
        classify_cfg = config.get_endpoint_config(HTTPMethod.POST, "/classify")
        assert "body_map" not in predict_cfg
        assert "body_map" not in classify_cfg

    @staticmethod
    def test_add_body_mapping() -> None:
        """Test add_body_mapping helper method"""
        config = APIHandlerConfig()
        assert config.body_map == {}

        # Add first mapping
        config.add_body_mapping("user_name", "$.user.name")
        assert config.body_map == {"user_name": "$.user.name"}

        # Add second mapping
        config.add_body_mapping("user_email", "$.user.contact.email")
        assert config.body_map == {
            "user_name": "$.user.name",
            "user_email": "$.user.contact.email",
        }

        # Add mapping with wildcard (multiple matches)
        config.add_body_mapping("item_ids", "$.items[*].id")
        assert config.body_map == {
            "user_name": "$.user.name",
            "user_email": "$.user.contact.email",
            "item_ids": "$.items[*].id",
        }

    @staticmethod
    def test_add_body_mapping_overwrites_existing(caplog) -> None:
        """Test that add_body_mapping overwrites existing mappings and logs warning"""

        config = APIHandlerConfig(body_map={"param": "$.old.path"})

        with caplog.at_level(logging.WARNING):
            config.add_body_mapping("param", "$.new.path")

        assert config.body_map == {"param": "$.new.path"}

        # Verify warning was logged
        assert any(
            "Overriding existing body mapping" in record.message
            for record in caplog.records
        )

    @staticmethod
    def test_remove_body_mapping() -> None:
        """Test remove_body_mapping helper method"""
        config = APIHandlerConfig(
            body_map={
                "user_name": "$.user.name",
                "user_email": "$.user.contact.email",
                "item_ids": "$.items[*].id",
            }
        )

        # Remove one mapping
        config.remove_body_mapping("user_email")
        assert config.body_map == {
            "user_name": "$.user.name",
            "item_ids": "$.items[*].id",
        }

        # Remove another
        config.remove_body_mapping("item_ids")
        assert config.body_map == {"user_name": "$.user.name"}

        # Remove last one
        config.remove_body_mapping("user_name")
        assert config.body_map == {}

    @staticmethod
    def test_remove_body_mapping_nonexistent() -> None:
        """Test that removing non-existent mapping doesn't raise error"""
        config = APIHandlerConfig(body_map={"param": "$.path"})
        config.remove_body_mapping("nonexistent")  # Should not raise
        assert config.body_map == {"param": "$.path"}

    @staticmethod
    def test_remove_body_mapping_when_empty() -> None:
        """Test removing mapping when body_map is empty"""
        config = APIHandlerConfig()
        assert config.body_map == {}
        config.remove_body_mapping("param")  # Should not raise
        assert config.body_map == {}

    @staticmethod
    def test_add_body_mapping_validates_jsonpath() -> None:
        """Test that add_body_mapping validates JSONPath expression"""
        import mlrun.errors

        config = APIHandlerConfig()

        # Should raise for invalid JSONPath syntax
        with pytest.raises(
            mlrun.errors.MLRunValueError,
            match=r"Invalid JSON path expression for parameter 'bad_param'",
        ):
            config.add_body_mapping("bad_param", "$.invalid[[[syntax")

        # Should not raise for valid JSONPath
        config.add_body_mapping("good_param", "$.valid.path")
        assert config.body_map == {"good_param": "$.valid.path"}


# ---------------------------------------------------------------------------
# _APIHandlerStep body_map integration tests
# ---------------------------------------------------------------------------
class TestAPIHandlerStepBodyMap:
    """Tests for body_map integration in _APIHandlerStep"""

    @staticmethod
    def test_invalid_jsonpath_raises_error_on_init() -> None:
        """Test that invalid JSONPath expression raises error during initialization"""
        import mlrun.errors

        config = APIHandlerConfig(body_map={"param": "$.invalid[[[syntax"})
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, "Test"
        )

        # Should raise during step initialization, not during request handling
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"Invalid JSON path expression for parameter 'param'",
        ):
            _APIHandlerStep(config=config)

    @staticmethod
    def _make_step_with_body_map(body_map, path="/predict"):
        """Helper to create an _APIHandlerStep with a config-level body_map"""
        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler(
            path, HTTPMethod.POST, APIHandlerAction.ALLOW, "Test"
        )
        return _APIHandlerStep(config=config)

    @staticmethod
    def test_body_map_transforms_event_body() -> None:
        """Test that body_map transforms the event body via JSONPath"""
        body_map = {
            "model_name": "$.request.model",
            "input_data": "$.request.data",
        }
        step = TestAPIHandlerStepBodyMap._make_step_with_body_map(body_map)

        event = MagicMock()
        event.method = HTTPMethod.POST
        event.path = "/predict"
        event.body = {
            "request": {
                "model": "my-model",
                "data": [1, 2, 3],
            },
            "metadata": {"trace_id": "abc"},
        }

        result = step.do(event)
        assert result.body == {"model_name": "my-model", "input_data": [1, 2, 3]}

    @staticmethod
    def test_body_map_missing_params_raises_error() -> None:
        """Test that missing body_map params raise MLRunUnprocessableEntityError"""
        body_map = {
            "name": "$.name",
            "missing": "$.nonexistent.path",
        }
        step = TestAPIHandlerStepBodyMap._make_step_with_body_map(body_map)

        event = MagicMock()
        event.method = HTTPMethod.POST
        event.path = "/predict"
        event.body = {"name": "test-model"}

        # Should raise MLRunUnprocessableEntityError since $.nonexistent.path has no matches
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError, match="matched nothing"
        ):
            step.do(event)

    def test_no_body_map_passes_event_through(self) -> None:
        """Test that without body_map the event passes through unchanged"""
        config = APIHandlerConfig()  # no body_map
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)

        step = _APIHandlerStep(config=config)

        event = MagicMock()
        event.method = HTTPMethod.POST
        event.path = "/predict"
        original_body = {"data": [1, 2, 3]}
        event.body = original_body

        result = step.do(event)
        assert result.body is original_body

    @staticmethod
    def test_body_map_non_dict_body_raises_error() -> None:
        """Test that non-dict body raises MLRunUnprocessableEntityError when body_map is configured"""
        body_map = {"param": "$.field"}
        step = TestAPIHandlerStepBodyMap._make_step_with_body_map(body_map)

        event = MagicMock()
        event.method = HTTPMethod.POST
        event.path = "/predict"
        event.body = "plain string body"

        # Should raise MLRunUnprocessableEntityError since body_map requires dict body
        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="body_map configured but request body is not a dict",
        ):
            step.do(event)

    def test_body_map_applies_to_all_endpoints(self) -> None:
        """Test that the same body_map is applied regardless of which endpoint matched"""
        body_map = {"input": "$.payload.data"}
        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)
        config.add_endpoint_handler(
            "/classify", HTTPMethod.POST, APIHandlerAction.ALLOW
        )

        step = _APIHandlerStep(config=config)

        # Test /predict
        event = MagicMock()
        event.method = HTTPMethod.POST
        event.path = "/predict"
        event.body = {"payload": {"data": [1, 2]}}
        result = step.do(event)
        assert result.body == {"input": [1, 2]}

        # Test /classify -- same body_map applies
        event2 = MagicMock()
        event2.method = HTTPMethod.POST
        event2.path = "/classify"
        event2.body = {"payload": {"data": "cats"}}
        result2 = step.do(event2)
        assert result2.body == {"input": "cats"}


# ---------------------------------------------------------------------------
# End-to-end mock-server tests
# ---------------------------------------------------------------------------
class TestBodyMapMockServer:
    """End-to-end tests for body_map with mock server"""

    @staticmethod
    def test_body_map_e2e() -> None:
        """Test body_map end-to-end through mock server"""

        def echo_handler(**kwargs):
            return kwargs

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map", kind="serving"),
        )

        body_map = {
            "model_name": "$.request.model",
            "input_data": "$.request.data",
        }

        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Prediction with body_map",
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={
                    "request": {
                        "model": "my-model",
                        "data": [1, 2, 3],
                    },
                    "extra_field": "ignored",
                },
            )
            assert resp == {"model_name": "my-model", "input_data": [1, 2, 3]}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_with_missing_fields_e2e() -> None:
        """Test body_map with missing fields raises KeyError"""

        def echo_handler(**kwargs):
            return kwargs

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-missing", kind="serving"),
        )

        body_map = {
            "name": "$.user.name",
            "email": "$.user.email",
            "phone": "$.user.phone",  # This will be missing
        }

        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler(
            "/register",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Register with body_map",
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            # Should raise since $.user.phone is missing
            with pytest.raises(RuntimeError, match="matched nothing"):
                server.test(
                    "/register",
                    method="POST",
                    body={
                        "user": {
                            "name": "Alice",
                            "email": "alice@example.com",
                            # "phone" is intentionally missing
                        }
                    },
                )
        finally:
            server.wait_for_completion()

    def test_endpoint_without_body_map_unaffected(self) -> None:
        """Test that endpoints without body_map still work normally"""
        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-no-body-map", kind="serving"),
        )

        config = APIHandlerConfig()  # no body_map
        config.add_endpoint_handler(
            "/health", HTTPMethod.GET, APIHandlerAction.ALLOW, "Health check"
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler="(event)").respond()

        server = fn.to_mock_server()
        try:
            resp = server.test("/health", method="GET", body="ping")
            assert resp == "ping"
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_same_for_multiple_endpoints_e2e() -> None:
        """Test that the same body_map is applied to all endpoints"""

        def echo_handler(**kwargs):
            return kwargs

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-multi", kind="serving"),
        )

        body_map = {"input": "$.payload.data"}

        config = APIHandlerConfig(body_map=body_map)
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)
        config.add_endpoint_handler(
            "/classify", HTTPMethod.POST, APIHandlerAction.ALLOW
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            resp1 = server.test(
                "/predict",
                method="POST",
                body={"payload": {"data": [1, 2]}},
            )
            assert resp1 == {"input": [1, 2]}

            resp2 = server.test(
                "/classify",
                method="POST",
                body={"payload": {"data": "cats"}},
            )
            assert resp2 == {"input": "cats"}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_kwargs_handler_e2e() -> None:
        """Test body_map unpacks as kwargs to a handler with named parameters"""

        def fun(book):
            return f"{book} - this is the book"

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-kwargs", kind="serving"),
        )

        config = APIHandlerConfig(body_map={"book": "$.age"})
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Kwargs handler",
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="process", handler=fun).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"firstName": "John", "lastName": "doe", "age": 26},
            )
            assert resp == "26 - this is the book"
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_multi_kwargs_handler_e2e() -> None:
        """Test body_map with multiple kwargs passed to handler"""

        def handler(name, age):
            return f"{name} is {age} years old"

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-multi-kwargs", kind="serving"),
        )

        config = APIHandlerConfig(body_map={"name": "$.firstName", "age": "$.age"})
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="process", handler=handler).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"firstName": "John", "lastName": "doe", "age": 26},
            )
            assert resp == "John is 26 years old"
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_nested_extraction_e2e() -> None:
        """Test body_map with nested field extraction via JSONPath"""

        def my_step(param1, param2):
            return f"Received: {param1} and {param2}"

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-nested-extraction", kind="serving"),
        )

        config = APIHandlerConfig(
            body_map={
                "param1": "$.field1",  # Extract "field1" and pass as param1
                "param2": "$.nested.field2",  # Extract nested field and pass as param2
            }
        )
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="process", handler=my_step).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={
                    "field1": "value1",
                    "nested": {"field2": "value2"},
                    "extra": "ignored",  # This won't be passed to the handler
                },
            )
            assert resp == "Received: value1 and value2"
        finally:
            server.wait_for_completion()


def test_api_handler_with_body_map_and_processing_step(rundb_mock):
    """Test API handler with body_map followed by a processing step in the graph."""

    def pass_through(arg1, arg2):
        """Handler that passes through the mapped values"""
        return {"arg1": arg1, "arg2": arg2}

    fn = cast(
        ServingRuntime,
        mlrun.new_function("test-func", kind="serving", image="mlrun/mlrun"),
    )

    # Configure API handler with body mapping
    config = APIHandlerConfig()
    config.add_body_mapping("arg1", "$.data.field1")
    config.add_body_mapping("arg2", "$.data.nested.field2")
    config.add_endpoint_handler(
        "/predict",
        HTTPMethod.POST,
        APIHandlerAction.ALLOW,
        "Predict with body_map",
    )
    fn.set_api_handler_config(config)

    # Build graph: handler -> processor
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="my-handler", handler=pass_through).to(
        class_name="tests.serving.test_body_map.PrefixStep",
        name="processor",
        prefix="Result",
    ).respond()

    server = fn.to_mock_server()
    try:
        # Test with request body
        resp = server.test(
            "/predict",
            method="POST",
            body={
                "data": {
                    "field1": "hello",
                    "nested": {"field2": "world"},
                },
                "metadata": "ignored",
            },
        )
        assert resp == "Result: arg1=hello, arg2=world"
    finally:
        server.wait_for_completion()
