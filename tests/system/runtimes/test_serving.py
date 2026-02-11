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
    ) -> mlrun.runtimes.ServingRuntime:
        """Create a basic serving function for testing."""
        # Create serving function using project (no external file needed)
        function = self.project.set_function(
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
        assert (
            function.spec.api_handler_config is not None
        ), "API handler config should be set in function spec"
        # Convert back to APIHandlerConfig for comparison
        spec_config = APIHandlerConfig.from_dict(function.spec.api_handler_config)
        assert (
            spec_config.endpoints == config.endpoints
        ), "API handler endpoints should match the config set"
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
