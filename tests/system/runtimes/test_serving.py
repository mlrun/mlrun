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

import pytest

import mlrun
import tests.system.base
from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig


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
        assert spec_config.endpoints == config.endpoints, (
            "API handler endpoints should match the config set"
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
            match=r"MLRunBadRequestError: Access forbidden to GET /api/v1/admin",
        ):
            function.invoke(
                path="/api/v1/admin", method="GET", body={"test": "restricted"}
            )

        self._logger.info("Forbidden API handler test passed")

    def test_api_handler_with_body_mapping(self) -> None:
        """Test API handler with body_map JSONPath extraction."""
        self._logger.info("Testing API handler with body_map functionality")

        # Create API handler config with body_map for JSONPath extraction
        config = APIHandlerConfig()

        config.add_body_mapping("user_name", "$.user.name")
        config.add_body_mapping("user_email", "$.user.contact.email")
        # Multiple matches return list
        config.add_body_mapping("book_titles", "$.purchases[*].title")

        config.add_endpoint_handler(
            "/api/v1/process",
            HTTPMethod.POST,
            APIHandlerAction.ALLOW,
            "Process endpoint with body mapping",
        )

        # Create serving function with handler source file using helper method
        function = self._create_serving_function(
            name="body-map-handler",
            api_config=config,
            func=str(self.assets_path / "body_map_handler.py"),
        )

        # Set up topology with handler that receives kwargs from body_map
        graph = function.set_topology("flow", engine="sync", exist_ok=True)
        graph.to(
            name="processor", handler="process_mapped_data"
        ).respond()  # Reference handler by name

        # Deploy the function
        self._logger.debug("Deploying serving function with body_map")
        function.deploy()

        # Test with request body that has nested structure
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

        # Verify the mapped values were extracted correctly
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
