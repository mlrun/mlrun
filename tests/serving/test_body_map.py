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

import pytest

import mlrun
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import ServingRuntime
from mlrun.serving.api_handler import _APIHandlerStep, _RequestContext
from mlrun.serving.endpoint_mapping import APIHandlerConfig, BodyMappings
from mlrun.serving.server import MockEvent


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
    """Tests for per-endpoint input_body_mappings on APIHandlerConfig."""

    def test_add_body_mapping_on_specific_endpoint(self) -> None:
        """input_body_mappings is scoped to its endpoint — other endpoints are unaffected."""
        bm = BodyMappings()
        bm.add_mapping("$.data", destination_path="input")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=bm,
        )
        config.add_endpoint_handler(
            "/classify", HTTPMethod.POST, APIHandlerAction.ALLOW
        )

        predict_ep = config.get_endpoint_config(HTTPMethod.POST, "/predict")
        classify_ep = config.get_endpoint_config(HTTPMethod.POST, "/classify")

        assert predict_ep.input_body_mappings is not None
        assert predict_ep.input_body_mappings.mappings[0]["destination_path"] == "input"
        assert classify_ep.input_body_mappings is None

    def test_add_mapping_same_source_overrides_destination(self, caplog) -> None:
        """Calling add_mapping twice with the same source_path overwrites the destination.

        The second call with the same source but different destination_path must win.
        get_endpoint_config must return the updated destination.
        """
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model_old")

        with caplog.at_level(logging.WARNING):
            bm.add_mapping("$.model", destination_path="model_new")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        ep = config.get_endpoint_config(HTTPMethod.POST, "/predict")
        mappings = ep.input_body_mappings.mappings
        # Only one entry — the second add_mapping replaced the first
        assert len(mappings) == 1
        assert mappings[0]["source_path"] == "$.model"
        assert mappings[0]["destination_path"] == "model_new"
        assert any(
            "Overriding existing body mapping: duplicate source path" in record.message
            for record in caplog.records
        )

    def test_add_mapping_same_destination_overrides_source(self, caplog) -> None:
        """Calling add_mapping twice with the same destination_path overwrites the source.

        The second call with the same destination but different source_path must win.
        """
        bm = BodyMappings()
        bm.add_mapping("$.model_old", destination_path="model")

        with caplog.at_level(logging.WARNING):
            bm.add_mapping("$.model_new", destination_path="model")

        mappings = bm.mappings
        assert len(mappings) == 1
        assert mappings[0]["source_path"] == "$.model_new"
        assert mappings[0]["destination_path"] == "model"
        assert any(
            "Overriding existing body mapping: duplicate destination path"
            in record.message
            for record in caplog.records
        )

    def test_invalid_jsonpath_raises_at_step_init(self) -> None:
        """Invalid JSONPath in input_body_mappings raises MLRunValueError at _APIHandlerStep init."""
        bm = BodyMappings()
        bm.mappings = [
            {
                "source_path": "$.invalid[[[syntax",
                "destination_path": "bad_param",
                "mandatory": False,
            }
        ]

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        with pytest.raises(
            mlrun.errors.MLRunValueError, match="Invalid JSONPath expression"
        ):
            _APIHandlerStep(config=config)

    @staticmethod
    def test_invalid_jsonpath_raises_at_add_mapping() -> None:
        """Invalid JSONPath in add_mapping raises MLRunValueError immediately."""
        bm = BodyMappings()
        with pytest.raises(
            mlrun.errors.MLRunValueError, match="Invalid JSON path expression"
        ):
            bm.add_mapping("$.invalid[[[syntax", destination_path="bad_param")

    @staticmethod
    def test_remove_body_mapping() -> None:
        """remove_mapping removes entries by destination_path."""
        bm = BodyMappings()
        bm.add_mapping("$.user.name", destination_path="user_name")
        bm.add_mapping("$.user.contact.email", destination_path="user_email")
        bm.add_mapping("$.items[*].id", destination_path="item_ids")

        bm.remove_mapping("user_email")
        assert [m["destination_path"] for m in bm.mappings] == ["user_name", "item_ids"]

        bm.remove_mapping("item_ids")
        assert [m["destination_path"] for m in bm.mappings] == ["user_name"]

        bm.remove_mapping("user_name")
        assert bm.mappings == []

    @staticmethod
    def test_remove_body_mapping_nonexistent() -> None:
        """remove_mapping is a no-op when destination_path is not found."""
        bm = BodyMappings()
        bm.add_mapping("$.path", destination_path="param")
        bm.remove_mapping("nonexistent")  # Should not raise
        assert len(bm.mappings) == 1
        assert bm.mappings[0]["destination_path"] == "param"

    @staticmethod
    def test_remove_body_mapping_when_empty() -> None:
        """remove_mapping is a no-op on an empty BodyMappings."""
        bm = BodyMappings()
        bm.remove_mapping("param")  # Should not raise
        assert bm.mappings == []

    @pytest.mark.parametrize(
        "mapping,missing_field",
        [
            ({"destination_path": "model", "mandatory": False}, "source_path"),
            ({"source_path": "$.model", "mandatory": False}, "destination_path"),
        ],
    )
    def test_mappings_setter_missing_required_field_raises(
        self, mapping: dict, missing_field: str
    ) -> None:
        """mappings setter raises MLRunInvalidArgumentError when source_path or destination_path is missing."""
        bm = BodyMappings()
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError, match=f"'{missing_field}'"
        ):
            bm.mappings = [mapping]


# ---------------------------------------------------------------------------
# _APIHandlerStep body_map integration tests
# ---------------------------------------------------------------------------
class TestAPIHandlerStepBodyMap:
    """Tests for input_body_mappings integration in _APIHandlerStep."""

    @staticmethod
    def _make_step(
        mappings: dict[str, str], path: str = "/predict"
    ) -> "_APIHandlerStep":
        """Build a step with a single endpoint that has the given source→dest mappings."""
        bm = BodyMappings()
        for dest, src in mappings.items():
            bm.add_mapping(src, destination_path=dest)
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            path, HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )
        return _APIHandlerStep(config=config)

    def test_body_map_transforms_event_body(self) -> None:
        """input_body_mappings extracts fields from the request body via JSONPath."""
        step = self._make_step(
            {
                "model_name": "$.request.model",
                "input_data": "$.request.data",
            }
        )

        event = MockEvent(
            body={
                "request": {"model": "my-model", "data": [1, 2, 3]},
                "metadata": "ignored",
            },
            method="POST",
            path="/predict",
        )

        result = step.do(event)
        assert isinstance(result.body, _RequestContext)
        assert result.body["model_name"] == "my-model"
        assert result.body["input_data"] == [1, 2, 3]
        assert "metadata" not in result.body

    def test_body_map_multi_match_returns_list(self) -> None:
        """A JSONPath that matches multiple nodes returns a list of values."""
        step = self._make_step({"roles": "$.messages[*].role"})

        event = MockEvent(
            body={
                "messages": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}]
            },
            method="POST",
            path="/predict",
        )

        result = step.do(event)
        assert isinstance(result.body, _RequestContext)
        assert result.body["roles"] == ["user", "assistant", "user"]

    def test_body_map_missing_params_skipped(self) -> None:
        """Missing JSONPath fields are silently skipped when mandatory=False (default)."""
        step = self._make_step(
            {
                "name": "$.name",
                "missing": "$.nonexistent.path",
            }
        )

        event = MockEvent(body={"name": "test-model"}, method="POST", path="/predict")

        result = step.do(event)
        assert isinstance(result.body, _RequestContext)
        assert result.body["name"] == "test-model"
        assert "missing" not in result.body

    def test_no_body_map_passes_event_through(self) -> None:
        """Endpoint with no input_body_mappings passes the body through unchanged."""
        config = APIHandlerConfig()
        config.add_endpoint_handler("/predict", HTTPMethod.POST, APIHandlerAction.ALLOW)
        step = _APIHandlerStep(config=config)

        original_body = {"data": [1, 2, 3]}
        event = MockEvent(body=original_body, method="POST", path="/predict")

        result = step.do(event)
        assert result.body is original_body

    def test_body_map_non_dict_body_skipped(self) -> None:
        """Non-dict body is passed through unchanged even when mappings are configured."""
        step = self._make_step({"param": "$.field"})

        event = MockEvent(body="plain string body", method="POST", path="/predict")

        result = step.do(event)
        assert result.body == "plain string body"

    def test_different_body_maps_per_endpoint(self) -> None:
        """Each endpoint has its own input_body_mappings — only the matched one applies."""
        predict_bm = BodyMappings()
        predict_bm.add_mapping("$.request.model", destination_path="model_name")

        classify_bm = BodyMappings()
        classify_bm.add_mapping("$.payload.data", destination_path="input")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=predict_bm,
        )
        config.add_endpoint_handler(
            "/classify",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=classify_bm,
        )
        step = _APIHandlerStep(config=config)

        shared_body = {"request": {"model": "gpt-4"}, "payload": {"data": "cats"}}

        predict_result = step.do(
            MockEvent(body=shared_body, method="POST", path="/predict")
        )
        assert isinstance(predict_result.body, _RequestContext)
        assert predict_result.body["model_name"] == "gpt-4"
        assert "input" not in predict_result.body

        classify_result = step.do(
            MockEvent(body=shared_body, method="POST", path="/classify")
        )
        assert isinstance(classify_result.body, _RequestContext)
        assert classify_result.body["input"] == "cats"
        assert "model_name" not in classify_result.body

    def test_different_http_methods_independent(self) -> None:
        """GET and POST on the same path have independent body mappings."""
        get_bm = BodyMappings()
        get_bm.add_mapping("$.query", destination_path="q")

        post_bm = BodyMappings()
        post_bm.add_mapping("$.model", destination_path="model")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
            input_body_mappings=get_bm,
        )
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=post_bm,
        )
        step = _APIHandlerStep(config=config)

        shared_body = {"query": "hello", "model": "gpt-4"}

        get_result = step.do(MockEvent(body=shared_body, method="GET", path="/predict"))
        assert isinstance(get_result.body, _RequestContext)
        assert get_result.body["q"] == "hello"
        assert "model" not in get_result.body

        post_result = step.do(
            MockEvent(body=shared_body, method="POST", path="/predict")
        )
        assert isinstance(post_result.body, _RequestContext)
        assert post_result.body["model"] == "gpt-4"
        assert "q" not in post_result.body


# ---------------------------------------------------------------------------
# End-to-end mock-server tests
# ---------------------------------------------------------------------------
class TestBodyMapMockServer:
    """End-to-end tests for input_body_mappings with mock server."""

    @staticmethod
    def test_body_map_e2e() -> None:
        """input_body_mappings extracts fields end-to-end through the mock server."""

        def echo_handler(body, **kwargs):
            return kwargs

        fn = cast(ServingRuntime, mlrun.new_function("test-body-map", kind="serving"))

        bm = BodyMappings()
        bm.add_mapping("$.request.model", destination_path="model_name")
        bm.add_mapping("$.request.data", destination_path="input_data")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Prediction with body_map",
            input_body_mappings=bm,
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
                    "request": {"model": "my-model", "data": [1, 2, 3]},
                    "extra_field": "ignored",
                },
            )
            assert resp == {"model_name": "my-model", "input_data": [1, 2, 3]}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_with_missing_fields_e2e() -> None:
        """Missing JSONPath fields are silently skipped end-to-end."""

        def echo_handler(body, **kwargs):
            return kwargs

        fn = cast(
            ServingRuntime, mlrun.new_function("test-body-map-missing", kind="serving")
        )

        bm = BodyMappings()
        bm.add_mapping("$.user.name", destination_path="name")
        bm.add_mapping("$.user.email", destination_path="email")
        bm.add_mapping("$.user.phone", destination_path="phone")  # will be missing

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/register",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Register with body_map",
            input_body_mappings=bm,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/register",
                method="POST",
                body={"user": {"name": "Alice", "email": "alice@example.com"}},
            )
            assert resp == {"name": "Alice", "email": "alice@example.com"}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_different_body_maps_per_endpoint_e2e() -> None:
        """Each endpoint uses its own input_body_mappings — only matched fields are extracted."""

        def echo_handler(body, **kwargs):
            return kwargs

        fn = cast(
            ServingRuntime, mlrun.new_function("test-body-map-multi", kind="serving")
        )

        predict_bm = BodyMappings()
        predict_bm.add_mapping("$.request.model", destination_path="model_name")

        classify_bm = BodyMappings()
        classify_bm.add_mapping("$.payload.data", destination_path="input")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=predict_bm,
        )
        config.add_endpoint_handler(
            "/classify",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=classify_bm,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        shared_body = {"request": {"model": "gpt-4"}, "payload": {"data": [1, 2]}}

        server = fn.to_mock_server()
        try:
            resp1 = server.test("/predict", method="POST", body=shared_body)
            assert resp1 == {"model_name": "gpt-4"}

            resp2 = server.test("/classify", method="POST", body=shared_body)
            assert resp2 == {"input": [1, 2]}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_multi_match_body_map_merge_e2e() -> None:
        """E2e: star + specific endpoint both have body maps — merged result reaches the handler."""

        def echo_handler(body, **kwargs):
            return kwargs

        fn = cast(
            ServingRuntime, mlrun.new_function("test-multi-match-merge", kind="serving")
        )

        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model")

        specific_bm = BodyMappings()
        specific_bm.add_mapping("$.temperature", destination_path="temperature")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/*", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=star_bm
        )
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=specific_bm,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            resp = server.test(
                "/predict",
                method="POST",
                body={"model": "gpt-4", "temperature": 0.7, "extra": "ignored"},
            )
            # star contributes "model", specific contributes "temperature", extra is dropped
            assert resp == {"model": "gpt-4", "temperature": 0.7}
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_body_map_kwargs_handler_e2e() -> None:
        """input_body_mappings unpacks as kwargs to a handler with named parameters."""

        def fun(body, book):
            return f"{book} - this is the book"

        fn = cast(
            ServingRuntime, mlrun.new_function("test-body-map-kwargs", kind="serving")
        )

        bm = BodyMappings()
        bm.add_mapping("$.age", destination_path="book")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Kwargs handler",
            input_body_mappings=bm,
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
        """Multiple mapped fields are unpacked as kwargs to the handler."""

        def handler(body, name, age):
            return f"{name} is {age} years old"

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-multi-kwargs", kind="serving"),
        )

        bm = BodyMappings()
        bm.add_mapping("$.firstName", destination_path="name")
        bm.add_mapping("$.age", destination_path="age")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
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
        """Nested JSONPath extraction works end-to-end through the mock server."""

        def my_step(first_arg, param1, param2):
            return f"Received: {first_arg}, {param1=} and {param2=}"

        fn = cast(
            ServingRuntime, mlrun.new_function("test-nested-extraction", kind="serving")
        )

        bm = BodyMappings()
        bm.add_mapping("$.field1", destination_path="param1")
        bm.add_mapping("$.nested.field2", destination_path="param2")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
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
                    "extra": "ignored",
                },
            )
            assert resp == (
                "Received: {'field1': 'value1', 'nested': {'field2': 'value2'}, "
                "'extra': 'ignored'}, param1='value1' and param2='value2'"
            )
        finally:
            server.wait_for_completion()

    @staticmethod
    def test_missing_mandatory_input_field_returns_422() -> None:
        """End-to-end: missing mandatory input field surfaces as HTTP 422 to the caller."""

        def echo_handler(body, **kwargs):
            return kwargs

        fn = cast(
            ServingRuntime,
            mlrun.new_function("test-body-map-422", kind="serving"),
        )

        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=bm,
        )
        fn.set_api_handler_config(config)

        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="echo", handler=echo_handler).respond()

        server = fn.to_mock_server()
        try:
            with pytest.raises(
                RuntimeError,
                match=r"failed \(422\):.*Mandatory field 'model' not found",
            ):
                server.test(
                    "/predict",
                    method="POST",
                    body={"messages": ["hello"]},  # 'model' missing
                )
        finally:
            server.wait_for_completion()


def test_api_handler_with_body_map_and_processing_step(rundb_mock):
    """Test API handler with input_body_mappings followed by a processing step in the graph."""

    def pass_through(body, arg1, arg2):
        return {"arg1": arg1, "arg2": arg2}

    fn = cast(
        ServingRuntime,
        mlrun.new_function("test-func", kind="serving", image="mlrun/mlrun"),
    )

    bm = BodyMappings()
    bm.add_mapping("$.data.field1", destination_path="arg1")
    bm.add_mapping("$.data.nested.field2", destination_path="arg2")

    config = APIHandlerConfig()
    config.add_endpoint_handler(
        "/predict",
        HTTPMethod.POST,
        APIHandlerAction.ALLOW,
        "Predict with body_map",
        input_body_mappings=bm,
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


# ---------------------------------------------------------------------------
# Per-endpoint BodyMappings — mandatory field behavior
# ---------------------------------------------------------------------------
class TestPerEndpointBodyMappings:
    """Unit tests for per-endpoint input_body_mappings: extraction and mandatory enforcement."""

    def test_mandatory_field_missing_raises(self) -> None:
        """mandatory=True raises MLRunUnprocessableEntityError when the field is absent from the body."""
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        step = _APIHandlerStep(config=config)
        event = MockEvent(body={"messages": ["hello"]}, method="POST", path="/predict")

        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="Mandatory field 'model' not found",
        ):
            step.do(event)

    @pytest.mark.parametrize("body", ["not-a-dict", None])
    def test_non_dict_body_with_mandatory_mapping_raises(self, body) -> None:
        """Non-dict body with a mandatory mapping raises MLRunUnprocessableEntityError (HTTP 422).

        When body_map has at least one mandatory field, the contract can't be satisfied
        without a dict body — so we fail fast rather than silently skip.
        """
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        step = _APIHandlerStep(config=config)
        event = MockEvent(body=body, method="POST", path="/predict")

        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match=r"Mandatory input body mappings configured but input body is not a dict",
        ):
            step.do(event)

    @pytest.mark.parametrize("body", ["not-a-dict", None])
    def test_non_dict_body_with_optional_mapping_silently_skips(self, body) -> None:
        """Non-dict body with only optional mappings is silently skipped (no error)."""
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=False)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        step = _APIHandlerStep(config=config)
        event = MockEvent(body=body, method="POST", path="/predict")

        # Should not raise — body mapping is silently skipped when body isn't a dict
        # and no mappings are mandatory.
        step.do(event)

    @pytest.mark.parametrize("mandatory", [True, False])
    def test_mapped_field_extracted(self, mandatory: bool) -> None:
        """Mapped field is extracted correctly regardless of mandatory flag when present.

        Only the declared destination ('model') must appear in the context.
        An unrelated body key ('extra_data') must NOT be extracted.
        """
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=mandatory)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=bm
        )

        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"model": "gpt-4", "extra_data": "ignored"},
            method="POST",
            path="/predict",
        )

        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["model"] == "gpt-4"
        assert "extra_data" not in result.body


# ---------------------------------------------------------------------------
# Hierarchical body map merging across star → exact/template endpoints
# ---------------------------------------------------------------------------
class TestBodyMapHierarchy:
    """Tests for hierarchical body map merging across star → exact/template endpoints."""

    # All tests send a request to /predict/1.
    # The sub-endpoint is parametrized as exact, template, or specific-star.
    _REQUEST_PATH = "/predict/1"
    _SUB_ENDPOINT_PATHS = [
        "/predict/1",  # exact
        "/predict/{item_id}",  # template — also extracts item_id="1"
        "/predict/*",  # specific star
    ]

    def _make_config(
        self,
        star_bm: BodyMappings | None,
        explicit_bm: BodyMappings | None,
        star_first: bool,
        explicit_path: str = "/predict/1",
    ) -> APIHandlerConfig:
        """Build config with a broad '/*' star and one explicit endpoint.

        Insertion order is controlled by star_first.
        """
        config = APIHandlerConfig()
        if star_first:
            config.add_endpoint_handler(
                "/*",
                HTTPMethod.POST,
                APIHandlerAction.ALLOW,
                input_body_mappings=star_bm,
            )
            config.add_endpoint_handler(
                explicit_path,
                HTTPMethod.POST,
                APIHandlerAction.ALLOW,
                input_body_mappings=explicit_bm,
            )
        else:
            config.add_endpoint_handler(
                explicit_path,
                HTTPMethod.POST,
                APIHandlerAction.ALLOW,
                input_body_mappings=explicit_bm,
            )
            config.add_endpoint_handler(
                "/*",
                HTTPMethod.POST,
                APIHandlerAction.ALLOW,
                input_body_mappings=star_bm,
            )
        return config

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    def test_star_body_map_inherited_when_sub_has_none(
        self, star_first: bool, explicit_path: str
    ) -> None:
        """Star body map is applied to sub-path when sub has no body mapping.

        For template paths, the path parameter is also extracted alongside the
        inherited body mapping field. Insertion order must not affect the result.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=None,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"model": "gpt-4", "extra": "ignored"},
            method="POST",
            path=self._REQUEST_PATH,
        )

        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["model"] == "gpt-4"
        assert "extra" not in result.body
        if explicit_path == "/predict/{item_id}":
            assert result.body["item_id"] == "1"

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    def test_star_mandatory_inherited_raises_when_field_missing(
        self, star_first: bool, explicit_path: str
    ) -> None:
        """Star mandatory mapping raises when field missing and sub has no body mapping.

        Insertion order must not affect the result.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=None,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"other": "value"}, method="POST", path=self._REQUEST_PATH
        )

        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="Mandatory field 'model' not found",
        ):
            step.do(event)

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    @pytest.mark.parametrize("explicit_dest", ["model", "model_renamed"])
    def test_explicit_optional_overrides_star_mandatory(
        self, star_first: bool, explicit_path: str, explicit_dest: str
    ) -> None:
        """More specific endpoint's mandatory=False overrides broad star's mandatory=True.

        When the field is missing, no error is raised because the explicit mapping wins.
        Covers both same-destination and renamed-destination cases.
        Insertion order must not affect the result.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model", mandatory=True)

        explicit_bm = BodyMappings()
        explicit_bm.add_mapping(
            "$.model", destination_path=explicit_dest, mandatory=False
        )

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=explicit_bm,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"other": "value"}, method="POST", path=self._REQUEST_PATH
        )

        # explicit's mandatory=False wins → no error, field missing → not extracted.
        # For template paths, path params still create a _RequestContext.
        result = step.do(event)
        assert "model" not in result.body
        assert explicit_dest not in result.body
        if explicit_path == "/predict/{item_id}":
            assert isinstance(result.body, _RequestContext)
            assert result.body["item_id"] == "1"
        else:
            # No params extracted at all → original body passed through unchanged
            assert result.body == {"other": "value"}

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    def test_explicit_mandatory_overrides_star_optional(
        self, star_first: bool, explicit_path: str
    ) -> None:
        """More specific endpoint's mandatory=True overrides broad star's mandatory=False.

        When the field is missing, an error IS raised because the explicit mapping wins.
        Insertion order must not affect the result.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model", mandatory=False)

        explicit_bm = BodyMappings()
        explicit_bm.add_mapping("$.model", destination_path="model", mandatory=True)

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=explicit_bm,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"other": "value"}, method="POST", path=self._REQUEST_PATH
        )

        with pytest.raises(
            mlrun.errors.MLRunUnprocessableEntityError,
            match="Mandatory field 'model' not found",
        ):
            step.do(event)

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    def test_star_and_sub_mappings_combined(
        self, star_first: bool, explicit_path: str
    ) -> None:
        """Star and sub-path each map a different field — both are extracted.

        Body has 3 keys: 'model' (mapped by star), 'temperature' (mapped by sub-path),
        and 'extra' (not mapped by either). Only 'model' and 'temperature' must appear
        in the context; 'extra' must not. For template paths, the path param is also
        extracted. Insertion order must not affect the result.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model")

        explicit_bm = BodyMappings()
        explicit_bm.add_mapping("$.temperature", destination_path="temperature")

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=explicit_bm,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"model": "gpt-4", "temperature": 0.7, "extra": "ignored"},
            method="POST",
            path=self._REQUEST_PATH,
        )

        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["model"] == "gpt-4"
        assert result.body["temperature"] == 0.7
        assert "extra" not in result.body
        if explicit_path == "/predict/{item_id}":
            assert result.body["item_id"] == "1"

    @pytest.mark.parametrize("star_first", [True, False])
    @pytest.mark.parametrize("explicit_path", _SUB_ENDPOINT_PATHS)
    def test_same_source_different_dest_specific_wins(
        self, star_first: bool, explicit_path: str
    ) -> None:
        """Same source_path on star and specific endpoint — only specific destination survives.

        Star maps $.model → model_star; specific maps $.model → model_specific.
        After merge, only model_specific must be present; model_star must be gone.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model_star")

        explicit_bm = BodyMappings()
        explicit_bm.add_mapping("$.model", destination_path="model_specific")

        config = self._make_config(
            star_bm=star_bm,
            explicit_bm=explicit_bm,
            star_first=star_first,
            explicit_path=explicit_path,
        )
        step = _APIHandlerStep(config=config)
        event = MockEvent(
            body={"model": "gpt-4"},
            method="POST",
            path=self._REQUEST_PATH,
        )

        result = step.do(event)

        assert isinstance(result.body, _RequestContext)
        assert result.body["model_specific"] == "gpt-4"
        assert "model_star" not in result.body

    def test_forbid_star_body_map_still_merged_when_specific_allows(self) -> None:
        """FORBID star body map is still merged when a more specific ALLOW endpoint matches.

        Action (ALLOW/FORBID) is taken from the most specific match only.
        Body maps from all matches — including FORBID ones — are always merged.
        """
        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model")

        specific_bm = BodyMappings()
        specific_bm.add_mapping("$.temperature", destination_path="temperature")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/*", HTTPMethod.POST, APIHandlerAction.FORBID, input_body_mappings=star_bm
        )
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=specific_bm,
        )
        step = _APIHandlerStep(config=config)

        event = MockEvent(
            body={"model": "gpt-4", "temperature": 0.7},
            method="POST",
            path="/predict",
        )
        result = step.do(event)

        # Most specific match is ALLOW → request goes through
        # Both star (FORBID) and specific (ALLOW) body maps are merged
        assert isinstance(result.body, _RequestContext)
        assert result.body["model"] == "gpt-4"
        assert result.body["temperature"] == 0.7
