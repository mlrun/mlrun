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

"""Unit tests for set_openai_frontend() and OpenAI body mappings."""

from http import HTTPMethod
from typing import cast

import mlrun
from mlrun.runtimes.nuclio.serving import ServingRuntime
from mlrun.serving.endpoint_mapping import APIHandlerConfig
from mlrun.serving.openai_mappings import ENDPOINT_CLASSES, OpenAIEndpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fn() -> ServingRuntime:
    return cast(ServingRuntime, mlrun.new_function("test-openai", kind="serving"))


def _config(fn: ServingRuntime) -> APIHandlerConfig:
    return APIHandlerConfig.from_dict(fn.spec.api_handler_config)


# ---------------------------------------------------------------------------
# Group 1 — Registry structure
# ---------------------------------------------------------------------------
class TestOpenAIRegistry:
    def test_all_groups_present(self) -> None:
        """Registry contains an entry for every OpenAIEndpoint value."""
        for group in OpenAIEndpoint:
            assert group in ENDPOINT_CLASSES


# ---------------------------------------------------------------------------
# Group 2 — set_openai_frontend() wiring
# ---------------------------------------------------------------------------
class TestSetOpenAIFrontend:
    def test_responses_only_registers_correct_endpoints(self) -> None:
        """set_openai_frontend([RESPONSES]) registers exactly the Responses endpoints."""
        fn = _make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = _config(fn)
        responses_endpoints = ENDPOINT_CLASSES[OpenAIEndpoint.RESPONSES].endpoints()

        for ep_def in responses_endpoints:
            ep = config.get_endpoint_config(ep_def.http_method, ep_def.path)
            assert ep is not None, (
                f"Expected {ep_def['http_method']} {ep_def['path']} to be registered"
            )

        assert len(config.endpoints) == len(responses_endpoints)

    def test_default_registers_all_groups(self) -> None:
        """set_openai_frontend() with no args registers all OpenAIEndpoint groups."""
        fn = _make_fn()
        fn.set_openai_frontend()

        config = _config(fn)
        for group in OpenAIEndpoint:
            for ep_def in ENDPOINT_CLASSES[group].endpoints():
                ep = config.get_endpoint_config(ep_def.http_method, ep_def.path)
                assert ep is not None, (
                    f"Expected {ep_def['http_method']} {ep_def['path']} to be registered"
                )

    def test_preserves_existing_config(self) -> None:
        """set_openai_frontend() merges into an existing APIHandlerConfig."""
        from mlrun.common.schemas.serving import APIHandlerAction
        from mlrun.serving.endpoint_mapping import APIHandlerConfig

        existing = APIHandlerConfig()
        existing.add_endpoint_handler("/health", HTTPMethod.GET, APIHandlerAction.ALLOW)
        fn = _make_fn()
        fn.set_api_handler_config(existing)

        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = _config(fn)
        assert config.get_endpoint_config(HTTPMethod.GET, "/health") is not None
        assert (
            config.get_endpoint_config(HTTPMethod.POST, "/responses/compact")
            is not None
        )
