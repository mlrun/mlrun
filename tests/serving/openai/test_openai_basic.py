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

"""Registry and set_openai_frontend() wiring tests."""

from http import HTTPMethod

from mlrun.serving.endpoint_mapping import APIHandlerConfig
from mlrun.serving.openai_mappings import ENDPOINT_CLASSES, OpenAIEndpoint
from tests.serving.openai.openai_common import get_config, make_fn

# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------


class TestOpenAIRegistry:
    def test_all_groups_present(self) -> None:
        """Registry contains an entry for every OpenAIEndpoint value."""
        for group in OpenAIEndpoint:
            assert group in ENDPOINT_CLASSES


# ---------------------------------------------------------------------------
# set_openai_frontend() wiring
# ---------------------------------------------------------------------------


class TestSetOpenAIFrontend:
    def test_responses_only_registers_correct_endpoints(self) -> None:
        """set_openai_frontend([RESPONSES]) registers exactly the Responses endpoints."""
        fn = make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = get_config(fn)
        responses_endpoints = ENDPOINT_CLASSES[OpenAIEndpoint.RESPONSES].endpoints()

        for ep in responses_endpoints:
            endpoint = config.get_endpoint_config(ep["http_method"], ep["path"])
            assert endpoint is not None, (
                f"Expected {ep['http_method']} {ep['path']} to be registered"
            )

        assert len(config.endpoints) == len(responses_endpoints)

    def test_default_registers_all_groups(self) -> None:
        """set_openai_frontend() with no args registers all OpenAIEndpoint groups."""
        fn = make_fn()
        fn.set_openai_frontend()

        config = get_config(fn)
        for group in OpenAIEndpoint:
            for ep in ENDPOINT_CLASSES[group].endpoints():
                endpoint = config.get_endpoint_config(ep["http_method"], ep["path"])
                assert endpoint is not None, (
                    f"Expected {ep['http_method']} {ep['path']} to be registered"
                )

    def test_preserves_existing_config(self) -> None:
        """set_openai_frontend() merges into an existing APIHandlerConfig."""
        from mlrun.common.schemas.serving import APIHandlerAction

        existing = APIHandlerConfig()
        existing.add_endpoint_handler("/health", HTTPMethod.GET, APIHandlerAction.ALLOW)
        fn = make_fn()
        fn.set_api_handler_config(existing)

        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = get_config(fn)
        assert config.get_endpoint_config(HTTPMethod.GET, "/health") is not None
        assert (
            config.get_endpoint_config(HTTPMethod.POST, "/responses/compact")
            is not None
        )
