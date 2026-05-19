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

"""Tests for set_openai_frontend() and OpenAI endpoint group body mappings."""

from http import HTTPMethod
from typing import cast

import pytest

import mlrun
from mlrun.runtimes.nuclio.serving import ServingRuntime
from mlrun.serving.endpoint_mapping import APIHandlerConfig
from mlrun.serving.openai_mappings import ENDPOINT_CLASSES, OpenAIEndpoint
from mlrun.serving.server import GraphServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fn() -> ServingRuntime:
    return cast(ServingRuntime, mlrun.new_function("test-openai", kind="serving"))


def _config(fn: ServingRuntime) -> APIHandlerConfig:
    return APIHandlerConfig.from_dict(fn.spec.api_handler_config)


def _make_mock_server(endpoint_group: OpenAIEndpoint, handler) -> GraphServer:
    fn = _make_fn()
    fn.set_openai_frontend([endpoint_group])
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="handler", handler=handler).respond()
    return fn.to_mock_server()


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
        fn = _make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = _config(fn)
        responses_endpoints = ENDPOINT_CLASSES[OpenAIEndpoint.RESPONSES].endpoints()

        for ep in responses_endpoints:
            endpoint = config.get_endpoint_config(ep["http_method"], ep["path"])
            assert endpoint is not None, (
                f"Expected {ep['http_method']} {ep['path']} to be registered"
            )

        assert len(config.endpoints) == len(responses_endpoints)

    def test_default_registers_all_groups(self) -> None:
        """set_openai_frontend() with no args registers all OpenAIEndpoint groups."""
        fn = _make_fn()
        fn.set_openai_frontend()

        config = _config(fn)
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
        fn = _make_fn()
        fn.set_api_handler_config(existing)

        fn.set_openai_frontend([OpenAIEndpoint.RESPONSES])

        config = _config(fn)
        assert config.get_endpoint_config(HTTPMethod.GET, "/health") is not None
        assert (
            config.get_endpoint_config(HTTPMethod.POST, "/responses/compact")
            is not None
        )


# ---------------------------------------------------------------------------
# Responses group — mock server tests
# ---------------------------------------------------------------------------

_COMPACT_FULL_RESPONSE = {
    "id": "resp_123",
    "object": "response.compaction",
    "created_at": 1234567890,
    "output": [{"type": "text", "text": "Hello"}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "extra_field": "should_be_filtered",
}


class TestResponsesGroupMock:
    """End-to-end mock-server tests for the Responses endpoint group."""

    # --- 1. Input filtering per endpoint ---

    def test_compact_filters_extra_input_and_output_fields(self) -> None:
        """POST /responses/compact: extra request fields filtered from input;
        extra graph response fields filtered from output."""
        captured: dict = {}

        def handler(body, **kwargs):
            captured.update(kwargs)
            return _COMPACT_FULL_RESPONSE

        server = _make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(
                "/responses/compact",
                method="POST",
                body={
                    "model": "gpt-4",
                    "input": "Hello",
                    "instructions": "Be helpful",
                    "previous_response_id": "resp_0",
                    "prompt_cache_key": "cache_key_1",
                    "prompt_cache_retention": True,
                    "service_tier": "default",
                    "extra_field": "should_be_filtered",
                },
            )
            # Extra field must not reach the handler
            assert "extra_field" not in captured
            # Mapped fields must be present
            assert captured["model"] == "gpt-4"
            assert captured["input"] == "Hello"
            assert captured["instructions"] == "Be helpful"
            assert captured["previous_response_id"] == "resp_0"
            assert captured["prompt_cache_key"] == "cache_key_1"
            assert captured["prompt_cache_retention"] is True
            assert captured["service_tier"] == "default"
            # Full output structure must be returned
            assert resp["id"] == "resp_123"
            assert resp["object"] == "response.compaction"
            assert resp["created_at"] == 1234567890
            assert resp["output"] == [{"type": "text", "text": "Hello"}]
            assert resp["usage"] == {"input_tokens": 10, "output_tokens": 5}
            assert "extra_field" not in resp
        finally:
            server.wait_for_completion()

    def test_delete_path_param_extracted_and_returns_correct_shape(self) -> None:
        """DELETE /responses/{response_id}: path param extracted; extra fields filtered; response matches spec."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return {"id": kwargs["response_id"], "object": "response", "deleted": True}

        server = _make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test("/responses/resp_123", method="DELETE")
            assert captured_kwargs.get("response_id") == "resp_123"
            assert resp["id"] == "resp_123"
            assert resp["deleted"] is True
            assert resp["object"] == "response"
        finally:
            server.wait_for_completion()

    # --- 2. Incomplete graph response raises (mandatory output fields) ---

    def test_compact_incomplete_response_raises(self) -> None:
        """POST /responses/compact: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = _make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(
                mlrun.errors.MLRunBadRequestError, match="Mandatory field"
            ):
                server.test(
                    "/responses/compact",
                    method="POST",
                    body={"model": "gpt-4"},
                )
        finally:
            server.wait_for_completion()

    # --- 3. Mandatory input check per endpoint ---

    def test_compact_missing_mandatory_model_raises(self) -> None:
        """POST /responses/compact: missing mandatory 'model' input field → error."""

        def handler(body, **kwargs):
            return _COMPACT_FULL_RESPONSE

        server = _make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/responses/compact",
                    method="POST",
                    body={"input": "Hello"},  # model omitted
                )
        finally:
            server.wait_for_completion()


# ---------------------------------------------------------------------------
# ChatCompletions group — placeholder (ML-12461)
# ---------------------------------------------------------------------------
class TestChatCompletionsGroupMock:
    """Mock tests for the ChatCompletions endpoint group (endpoints TBD — ML-12461)."""

    def test_no_endpoints_registered(self) -> None:
        """ChatCompletions group has no endpoints yet."""
        fn = _make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.CHAT_COMPLETIONS])
        assert len(_config(fn).endpoints) == 0


# ---------------------------------------------------------------------------
# Audio group — placeholder (ML-12461)
# ---------------------------------------------------------------------------
class TestAudioGroupMock:
    """Mock tests for the Audio endpoint group (endpoints TBD — ML-12461)."""

    def test_no_endpoints_registered(self) -> None:
        """Audio group has no endpoints yet."""
        fn = _make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.AUDIO])
        assert len(_config(fn).endpoints) == 0


# ---------------------------------------------------------------------------
# Images group — placeholder (ML-12461)
# ---------------------------------------------------------------------------
class TestImagesGroupMock:
    """Mock tests for the Images endpoint group (endpoints TBD — ML-12461)."""

    def test_no_endpoints_registered(self) -> None:
        """Images group has no endpoints yet."""
        fn = _make_fn()
        fn.set_openai_frontend([OpenAIEndpoint.IMAGES])
        assert len(_config(fn).endpoints) == 0
