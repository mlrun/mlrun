# Copyright 2026 Iguazio
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

import time
from http import HTTPMethod

import httpx
import pytest

import mlrun
import tests.system.base
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.serving.endpoint_mapping import APIHandlerConfig, BodyMappings
from mlrun.serving.openai_mappings import OpenAIEndpoint


def assert_endpoint_configs_equal(
    actual: dict, expected: dict, context: str = ""
) -> None:
    """Assert two endpoint dicts are equal by comparing all EndpointConfig fields."""
    prefix = f"{context}: " if context else ""
    assert set(actual.keys()) == set(expected.keys()), (
        f"{prefix}endpoint keys differ: {set(actual.keys())} != {set(expected.keys())}"
    )
    for key in expected:
        a, e = actual[key], expected[key]
        assert a.path == e.path, f"{prefix}[{key}] path: {a.path!r} != {e.path!r}"
        assert a.http_method == e.http_method, (
            f"{prefix}[{key}] http_method: {a.http_method} != {e.http_method}"
        )
        assert a.action == e.action, f"{prefix}[{key}] action: {a.action} != {e.action}"
        assert a.description == e.description, (
            f"{prefix}[{key}] description: {a.description!r} != {e.description!r}"
        )
        a_bm = a.input_body_mappings.to_dict() if a.input_body_mappings else None
        e_bm = e.input_body_mappings.to_dict() if e.input_body_mappings else None
        assert a_bm == e_bm


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestServingAPIHandler(tests.system.base.TestMLRunSystem):
    """System tests for serving function API handler functionality."""

    project_name = "serving-api-handler"
    image: str | None = None

    def _create_serving_function(
        self,
        name: str,
        api_config: APIHandlerConfig | None = None,
        func: str | None = None,
    ) -> mlrun.runtimes.ServingRuntime:
        """Create a basic serving function for testing."""
        # Create serving function using project (no external file needed)
        function = self.project.set_function(
            func=func,
            name=name,
            kind="serving",
            image=self.image,
        )

        # Set API handler config if provided
        if api_config:
            function.set_api_handler_config(api_config)

        # Set up basic serving topology with inline handler like unit test
        graph = function.set_topology("flow", engine="sync")
        graph.to(name="echo", handler="(event)").respond()

        return function

    def test_api_handler_allowed(self) -> None:
        """Test that allowed API handlers work correctly (allow case)."""
        self._logger.info("Testing allowed API handler functionality")

        # Create API handler config that allows specific endpoint
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Prediction endpoint",
        )

        # Create and deploy serving function with allowed handler
        function = self._create_serving_function(
            name="allowed-api-handler", api_config=config
        )

        # Check that the API handler config is set correctly in the function spec
        assert function.spec.api_handler_config is not None, (
            "API handler config should be set in function spec"
        )
        # Convert back to APIHandlerConfig for comparison
        spec_config = APIHandlerConfig.from_dict(function.spec.api_handler_config)
        assert_endpoint_configs_equal(
            spec_config.endpoints,
            config.endpoints,
            context="API handler endpoints should match the config set",
        )
        self._logger.debug(
            "API handler config correctly set in function spec",
            endpoints=spec_config.endpoints,
        )

        # Deploy the function
        self._logger.debug("Deploying serving function with allowed handler")
        function.deploy()

        # Test the allowed API endpoint using function.invoke
        self._logger.debug("Testing allowed API handler endpoint")
        response = function.invoke(path="/api/v1/predict", body={"test": "data"})

        # Verify response - inline handler just echoes the event
        assert response is not None, "Handler should return a response"
        self._logger.info("Allowed API handler test passed")

    def test_api_handler_forbidden(self) -> None:
        """Test that forbidden API handlers are properly restricted (forbid case)."""
        self._logger.info("Testing forbidden API handler functionality")

        # Create API handler config that forbids specific endpoint
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/admin",
            HTTPMethod.GET,
            APIHandlerAction.FORBID,
            "Admin endpoint blocked",
        )

        # Create serving function with restricted handler
        function = self._create_serving_function(
            name="forbidden-api-handler", api_config=config
        )

        # Deploy the function
        self._logger.debug(
            "Deploying serving function with forbidden API handler config"
        )
        function.deploy()

        # Test the forbidden API endpoint - this should raise an error
        self._logger.debug("Testing forbidden API handler endpoint")
        with pytest.raises(
            RuntimeError,
            match=r"MLRunAccessDeniedError: Access forbidden to GET /api/v1/admin",
        ):
            function.invoke(
                path="/api/v1/admin", method="GET", body={"test": "restricted"}
            )

        self._logger.info("Forbidden API handler test passed")

    @pytest.mark.parametrize("engine", ["sync", "async"])
    def test_api_handler_with_body_mapping(self, engine: str) -> None:
        """Test API handler with per-endpoint input_body_mappings JSONPath extraction."""
        self._logger.info("Testing API handler with body_map functionality")

        bm = BodyMappings()
        bm.add_mapping("$.user.name", destination_path="user_name")
        bm.add_mapping("$.user.contact.email", destination_path="user_email")
        bm.add_mapping("$.purchases[*].title", destination_path="book_titles")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/process",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Process endpoint with body mapping",
            input_body_mappings=bm,
        )

        function = self._create_serving_function(
            name="body-map-handler",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )

        graph = function.set_topology("flow", engine=engine, exist_ok=True)
        graph.to(name="processor", handler="process_mapped_data").respond()

        self._logger.debug("Deploying serving function with body_map")
        function.deploy()

        test_body = {
            "user": {
                "name": "Alice Smith",
                "contact": {"email": "alice@example.com", "phone": "+1234567890"},
            },
            "purchases": [
                {"title": "MLOps Handbook", "price": 29.99},
                {"title": "Python Guide", "price": 19.99},
                {"title": "Data Science Intro", "price": 39.99},
            ],
            "timestamp": "2026-02-11T10:00:00Z",
        }

        self._logger.debug("Testing body_map with nested JSONPath extraction")
        response = function.invoke(path="/api/v1/process", body=test_body)

        assert response is not None, "Handler should return a response"
        assert response["name"] == "Alice Smith", "user_name should be extracted"
        assert response["email"] == "alice@example.com", (
            "user_email should be extracted from nested path"
        )
        assert response["titles"] == [
            "MLOps Handbook",
            "Python Guide",
            "Data Science Intro",
        ], "book_titles should extract multiple matches as list"
        assert response["count"] == 3, "count should match number of books"

        self._logger.info("Body mapping API handler test passed")

    def test_body_map_merge_and_method_isolation(self) -> None:
        """Test two scenarios in one deployment:

        1. Hierarchical merge: star /*  maps $.model, specific /api/v1/predict maps $.temperature.
           A POST to /api/v1/predict receives both fields merged.
        2. HTTP method isolation: GET /api/v1/predict has its own body map ($.query → q).
           A GET request only receives q, not model or temperature.
        """
        self._logger.info("Testing body map inheritance and HTTP method isolation")

        star_bm = BodyMappings()
        star_bm.add_mapping("$.model", destination_path="model")

        post_bm = BodyMappings()
        post_bm.add_mapping("$.temperature", destination_path="temperature")

        get_bm = BodyMappings()
        get_bm.add_mapping("$.query", destination_path="q")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/*", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=star_bm
        )
        config.add_endpoint_handler(
            "/api/v1/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=post_bm,
        )
        config.add_endpoint_handler(
            "/api/v1/predict",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
            input_body_mappings=get_bm,
        )

        function = self._create_serving_function(
            name="bm-inherit-method",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )
        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="echo", handler="echo_kwargs").respond()

        self._logger.debug(
            "Deploying function for inheritance and method isolation test"
        )
        function.deploy()

        # POST: star contributes model, specific contributes temperature — both merged
        post_response = function.invoke(
            path="/api/v1/predict",
            method="POST",
            body={"model": "gpt-4", "temperature": 0.7, "extra": "ignored"},
        )
        assert post_response["model"] == "gpt-4", "model from star bm should be present"
        assert post_response["temperature"] == 0.7, (
            "temperature from specific bm should be present"
        )
        assert "extra" not in post_response, "unmapped field should be dropped"
        assert "q" not in post_response, (
            "GET-only field should not appear in POST response"
        )

        # GET: only q should be extracted, model and temperature must not appear
        get_response = function.invoke(
            path="/api/v1/predict",
            method="GET",
            body={"query": "hello", "model": "gpt-4", "temperature": 0.7},
        )
        assert get_response["q"] == "hello", "query mapped to q should be extracted"
        assert "model" not in get_response, (
            "POST-only field should not appear in GET response"
        )
        assert "temperature" not in get_response, (
            "POST-only field should not appear in GET response"
        )

        self._logger.info("Body map inheritance and HTTP method isolation test passed")

    def test_api_handler_mandatory_field_missing_returns_422(self) -> None:
        """Test that a missing mandatory field returns HTTP 422."""
        self._logger.info("Testing mandatory field enforcement")

        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=True)
        bm.add_mapping("$.temperature", destination_path="temperature", mandatory=False)

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/v1/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            input_body_mappings=bm,
        )

        function = self._create_serving_function(
            name="mandatory-bm",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )
        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="echo", handler="echo_kwargs").respond()

        self._logger.debug("Deploying function for mandatory field test")
        function.deploy()

        # Missing mandatory field → expect 422
        with pytest.raises(RuntimeError, match="422"):
            function.invoke(
                path="/api/v1/predict",
                method="POST",
                body={"temperature": 0.7},  # model is missing
            )

        # Optional field missing → no error, only temperature extracted
        response = function.invoke(
            path="/api/v1/predict",
            method="POST",
            body={"model": "gpt-4"},  # temperature is missing but optional
        )
        assert response["model"] == "gpt-4"
        assert "temperature" not in response

        self._logger.info("Mandatory field enforcement test passed")

    def test_api_handler_with_path_and_query_params(self) -> None:
        """Test API handler with path parameters and query parameters."""
        self._logger.info("Testing API handler with path and query parameters")

        # Create API handler config with path template
        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/api/items/{category}/{item_id}",
            HTTPMethod.GET,
            APIHandlerAction.ALLOW,
            "Path and query params endpoint",
        )

        # Create serving function with handler source file
        function = self._create_serving_function(
            name="path-query-handler",
            api_config=config,
            func=str(self.assets_path / "path_query_handler.py"),
        )

        # Set up topology with handler that receives path and query params
        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="processor", handler="process_path_and_query_params").respond()

        # Deploy the function
        self._logger.debug("Deploying serving function with path and query params")
        function.deploy()

        # Test with path params and repeated query params
        self._logger.debug("Testing path params and repeated query params")
        response = function.invoke(
            path="/api/items/electronics/laptop-123?tags=new&tags=featured&tags=sale&limit=10",
            method="GET",
        )

        # Verify the path and query params were extracted correctly
        assert response is not None, "Handler should return a response"
        assert response["category"] == "electronics", (
            "category path param should be extracted"
        )
        assert response["item_id"] == "laptop-123", (
            "item_id path param should be extracted"
        )
        assert response["limit"] == "10", "limit query param should be string"
        # See NUC-7459 - multiple matches should be returned as list
        # assert response["tags"] == ["new", "featured", "sale"], "tags query param should be list"
        # assert response["tags_count"] == 3, "tags_count should match number of tags"

        self._logger.info("Path and query params API handler test passed")

    def test_api_handler_wildcard_path(self) -> None:
        """Test API handler with a wildcard '*' path pattern."""
        self._logger.info("Testing API handler with wildcard star path")

        config = APIHandlerConfig(include_url_info=True)
        config.add_endpoint_handler(
            "/api/wildcard/*",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Wildcard catch-all endpoint",
        )

        function = self._create_serving_function(
            name="wildcard-handler",
            api_config=config,
            func=str(self.assets_path / "wildcard_path_handler.py"),
        )

        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="handler", handler="handle_wildcard").respond()

        self._logger.debug("Deploying serving function with wildcard path")
        function.deploy()
        time.sleep(5)  # Wait for deployment to complete

        # Verify a nested path under the wildcard is routed correctly
        self._logger.debug("Invoking /api/wildcard/v1/data")
        response = function.invoke(
            path="/api/wildcard/v1/data",
            method="POST",
            body={},
        )
        assert response is not None, "Handler should return a response"
        assert response["matched_path"] == "/api/wildcard/v1/data", (
            "mlrun_request_path should reflect the exact request path"
        )

        # Verify a different nested path also routes correctly
        self._logger.debug("Invoking /api/wildcard/users/42")
        response2 = function.invoke(
            path="/api/wildcard/users/42",
            method="POST",
            body={},
        )
        assert response2["matched_path"] == "/api/wildcard/users/42", (
            "mlrun_request_path should reflect the request path for a different sub-path"
        )

        self._logger.info("Wildcard path API handler test passed")

    def test_output_body_mapping(self) -> None:
        """Test output_body_mappings reshapes the graph response on a live deployment.

        Input:  input_body_mappings extracts $.model, $.temperature, $.debug_flag from request.
        Graph:  echo_kwargs returns them as-is: {"input_model", "input_temperature", "input_debug_flag"}.
        Output: output_body_mappings selects and renames only model and temperature.
                $.input_model       → output_model       (rename)
                $.input_temperature → output_temperature  (rename)
                $.nonexistent       → output_extra        (optional — not in response → None)
                input_debug_flag is intentionally not mapped — verifies output drops unmapped fields.
        """
        self._logger.info("Testing output body mapping on live deployment")

        # Input: extract three fields; input_debug_flag will NOT appear in the output mapping,
        # proving that output_body_mappings fully controls what the caller receives.
        in_bm = BodyMappings()
        in_bm.add_mapping("$.model", destination_path="input_model")
        in_bm.add_mapping("$.temperature", destination_path="input_temperature")
        in_bm.add_mapping("$.debug_flag", destination_path="input_debug_flag")

        # Output: reshape the graph response — only declare what the caller should receive.
        # input_debug_flag is deliberately omitted here to show unmapped fields are dropped.
        out_bm = BodyMappings()
        out_bm.add_mapping("$.input_model", destination_path="output_model")
        out_bm.add_mapping("$.input_temperature", destination_path="output_temperature")
        out_bm.add_mapping(
            "$.nonexistent", destination_path="output_extra"
        )  # optional — will be None

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Predict with output body mapping",
            input_body_mappings=in_bm,
            output_body_mappings=out_bm,
        )

        function = self._create_serving_function(
            name="output-body-map-handler",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )

        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="echo", handler="echo_kwargs").respond()

        self._logger.debug("Deploying serving function with output body mapping")
        function.deploy()

        self._logger.debug("Invoking /predict")
        response = function.invoke(
            path="/predict",
            body={"model": "gpt-4", "temperature": 0.7, "debug_flag": True},
        )

        assert response == {
            "output_model": "gpt-4",
            "output_temperature": 0.7,
            "output_extra": None,
            # input_debug_flag is not here — output mapping dropped it
        }

        self._logger.info("Output body mapping test passed")

    def test_output_body_mapping_mandatory_missing_returns_422(self) -> None:
        """Missing mandatory output field raises HTTP 422."""
        out_bm = BodyMappings()
        out_bm.add_mapping(
            "$.nonexistent", destination_path="output_result", mandatory=True
        )

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/predict",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            output_body_mappings=out_bm,
        )

        function = self._create_serving_function(
            name="output-mandatory-handler",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )
        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(name="echo", handler="echo_kwargs").respond()
        function.deploy()

        with pytest.raises(
            RuntimeError,
            match=r"bad function response 422:.*MLRunUnprocessableEntityError.*Mandatory field 'output_result' not found",  # noqa: E501
        ):
            function.invoke(path="/predict", body={"model": "gpt-4"})

        self._logger.info("Output mandatory missing field test passed")

    # ---------------------------------------------------------------------------
    # OpenAI frontend tests (set_openai_frontend)
    # ---------------------------------------------------------------------------

    def test_chat_completions_create(self) -> None:
        """POST /chat/completions via the real OpenAI SDK.

        Verifies end-to-end: routing, input body mapping, output body mapping
        (extra_field filtering), and that the SDK parses the response into a
        typed ChatCompletion object.
        """
        openai = pytest.importorskip("openai")

        function = self.project.set_function(
            func=str(self.assets_path / "openai_serving_handler.py"),
            name="openai-chat-completions",
            kind="serving",
            image=self.image,
        )
        function.set_openai_frontend([OpenAIEndpoint.CHAT_COMPLETIONS])
        graph = function.set_topology("flow", engine="sync")
        graph.to(name="handler", handler="chat_completion_handler").respond()

        self._logger.debug("Deploying OpenAI chat completions serving function")
        function.deploy()

        client = openai.OpenAI(
            base_url=function.get_url(),
            api_key="dummy",
            http_client=httpx.Client(verify=mlrun.mlconf.httpdb.http.verify),
        )

        self._logger.debug("Calling POST /chat/completions via OpenAI SDK")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(response, openai.types.chat.ChatCompletion), (
            "SDK should return a ChatCompletion instance"
        )
        assert response.id == "chatcmpl_system_test_123"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.model == "gpt-4"
        assert response.service_tier == "default"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].logprobs is None
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hello from MLRun!"

        self._logger.info("OpenAI chat completions create system test passed")

    def test_responses_create(self) -> None:
        """POST /responses via the real OpenAI SDK.

        Verifies end-to-end: routing, input body mapping, output body mapping
        (extra_field filtering), and that the SDK parses the response into a
        typed Response object.
        """
        openai = pytest.importorskip("openai")

        function = self.project.set_function(
            func=str(self.assets_path / "openai_serving_handler.py"),
            name="openai-responses",
            kind="serving",
            image=self.image,
        )
        function.set_openai_frontend([OpenAIEndpoint.RESPONSES])
        graph = function.set_topology("flow", engine="sync")
        graph.to(name="handler", handler="response_handler").respond()

        self._logger.debug("Deploying OpenAI responses serving function")
        function.deploy()

        client = openai.OpenAI(
            base_url=function.get_url(),
            api_key="dummy",
            http_client=httpx.Client(verify=mlrun.mlconf.httpdb.http.verify),
        )

        self._logger.debug("Calling POST /responses via OpenAI SDK")
        response = client.responses.create(
            model="gpt-4",
            input="Hello",
        )

        assert isinstance(response, openai.types.responses.Response), (
            "SDK should return a Response instance"
        )
        assert response.id == "resp_system_test_123"
        assert response.object == "response"
        assert response.created_at == 1741476542
        assert response.status == "completed"
        assert response.completed_at == 1741476543
        assert response.model == "gpt-4"
        assert response.error is None
        assert response.incomplete_details is None
        assert response.instructions is None
        assert response.max_output_tokens is None
        assert response.parallel_tool_calls is True
        assert response.previous_response_id is None
        assert response.reasoning.effort is None
        assert response.temperature == 1.0
        assert response.tool_choice == "auto"
        assert response.tools == []
        assert response.top_p == 1.0
        assert response.truncation == "disabled"
        assert response.store is True
        assert response.usage.input_tokens == 36
        assert response.usage.output_tokens == 87
        assert response.usage.total_tokens == 123
        assert response.metadata == {}
        assert len(response.output) == 1
        assert response.output[0].role == "assistant"
        assert response.output[0].content[0].text == "Hello from MLRun!"

        self._logger.info("OpenAI responses create system test passed")

    def test_chat_completions_missing_mandatory_output_raises(self) -> None:
        """POST /chat/completions — handler omits mandatory 'choices' output field → error."""
        openai = pytest.importorskip("openai")

        function = self.project.set_function(
            func=str(self.assets_path / "openai_serving_handler.py"),
            name="openai-chat-missing-mandatory",
            kind="serving",
            image=self.image,
        )
        function.set_openai_frontend([OpenAIEndpoint.CHAT_COMPLETIONS])
        graph = function.set_topology("flow", engine="sync")
        graph.to(
            name="handler", handler="chat_completion_handler_missing_mandatory"
        ).respond()

        self._logger.debug(
            "Deploying chat completions function with incomplete handler"
        )
        function.deploy()

        client = openai.OpenAI(
            base_url=function.get_url(),
            api_key="dummy",
            http_client=httpx.Client(verify=mlrun.mlconf.httpdb.http.verify),
        )

        with pytest.raises(openai.APIStatusError) as exc_info:
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.status_code == 422
        assert (
            "MLRunUnprocessableEntityError: Failed to process output body mapping: "
            "Mandatory field 'choices' not found in body" in str(exc_info.value)
        )

        self._logger.info("Chat completions missing mandatory output field test passed")

    def test_responses_missing_mandatory_output_raises(self) -> None:
        """POST /responses — handler omits mandatory 'id' output field → error."""
        openai = pytest.importorskip("openai")

        function = self.project.set_function(
            func=str(self.assets_path / "openai_serving_handler.py"),
            name="openai-responses-missing-mandatory",
            kind="serving",
            image=self.image,
        )
        function.set_openai_frontend([OpenAIEndpoint.RESPONSES])
        graph = function.set_topology("flow", engine="sync")
        graph.to(name="handler", handler="response_handler_missing_mandatory").respond()

        self._logger.debug("Deploying responses function with incomplete handler")
        function.deploy()

        client = openai.OpenAI(
            base_url=function.get_url(),
            api_key="dummy",
            http_client=httpx.Client(verify=mlrun.mlconf.httpdb.http.verify),
        )

        with pytest.raises(openai.APIStatusError) as exc_info:
            client.responses.create(
                model="gpt-4",
                input="Hello",
            )
        assert exc_info.value.status_code == 422
        assert (
            "MLRunUnprocessableEntityError: Failed to process output body mapping: "
            "Mandatory field 'id' not found in body" in str(exc_info.value)
        )

        self._logger.info("Responses missing mandatory output field test passed")
