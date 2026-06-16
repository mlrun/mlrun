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

"""Mock-server tests for the ChatCompletions endpoint group."""

import pytest

from mlrun.serving.openai_mappings import OpenAIEndpoint
from tests.serving.assets.openai_fixtures import (
    CHAT_EXPECTED_KWARGS,
    CHAT_EXPECTED_RESPONSE,
    CHAT_HANDLER_RESPONSE,
    CHAT_REQUEST_BODY,
    COMPLETION_ID,
    DELETE_CHAT_HANDLER_RESPONSE,
    LIST_CHAT_EXPECTED_RESPONSE,
    LIST_CHAT_HANDLER_RESPONSE,
    LIST_MESSAGES_EXPECTED_RESPONSE,
    LIST_MESSAGES_HANDLER_RESPONSE,
    UPDATE_CHAT_EXPECTED_KWARGS,
    UPDATE_CHAT_REQUEST_BODY,
)
from tests.serving.openai.openai_common import make_mock_server


class TestChatCompletionsGroupMock:
    """End-to-end mock-server tests for the ChatCompletions endpoint group."""

    # ---------------------------------------------------------------------------
    # POST /chat/completions
    # ---------------------------------------------------------------------------

    def test_chat_completions_filters_extra_input_and_output_fields(self) -> None:
        """POST /chat/completions: extra request fields filtered from input;
        extra graph response fields filtered from output."""
        captured: dict = {}

        def handler(body, **kwargs):
            captured.update(kwargs)
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test(
                "/chat/completions", method="POST", body=CHAT_REQUEST_BODY
            )
            assert "extra_field" not in captured
            for key, value in CHAT_EXPECTED_KWARGS.items():
                assert captured[key] == value, f"kwargs[{key!r}] mismatch"
            assert "extra_field" not in resp
            for key, value in CHAT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_chat_completions_missing_mandatory_messages_raises(self) -> None:
        """POST /chat/completions: missing mandatory 'messages' input field → error."""

        def handler(body, **kwargs):
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/chat/completions",
                    method="POST",
                    body={"model": "gpt-4"},  # messages omitted
                )
        finally:
            server.wait_for_completion()

    def test_chat_completions_missing_mandatory_model_raises(self) -> None:
        """POST /chat/completions: missing mandatory 'model' input field → error."""

        def handler(body, **kwargs):
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/chat/completions",
                    method="POST",
                    body={
                        "messages": [{"role": "user", "content": "Hi"}]
                    },  # model omitted
                )
        finally:
            server.wait_for_completion()

    def test_chat_completions_incomplete_response_raises(self) -> None:
        """POST /chat/completions: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/chat/completions",
                    method="POST",
                    body={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "model": "gpt-4",
                    },
                )
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # GET /chat/completions/{completion_id}
    # ---------------------------------------------------------------------------

    def test_get_path_param_extracted_and_returns_chat_completion(self) -> None:
        """GET /chat/completions/{completion_id}: path param extracted; extra output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test(f"/chat/completions/{COMPLETION_ID}", method="GET")
            assert captured_kwargs.get("completion_id") == COMPLETION_ID
            assert "extra_field" not in resp
            for key, value in CHAT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_get_incomplete_response_raises(self) -> None:
        """GET /chat/completions/{completion_id}: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/chat/completions/{COMPLETION_ID}", method="GET")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # POST /chat/completions/{completion_id}
    # ---------------------------------------------------------------------------

    def test_update_filters_extra_input_and_output_fields(self) -> None:
        """POST /chat/completions/{completion_id}: path param extracted; extra input/output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test(
                f"/chat/completions/{COMPLETION_ID}",
                method="POST",
                body=UPDATE_CHAT_REQUEST_BODY,
            )
            assert captured_kwargs.get("completion_id") == COMPLETION_ID
            assert "extra_field" not in captured_kwargs
            for key, value in UPDATE_CHAT_EXPECTED_KWARGS.items():
                assert captured_kwargs[key] == value, f"kwargs[{key!r}] mismatch"
            assert "extra_field" not in resp
            for key, value in CHAT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_update_missing_mandatory_metadata_raises(self) -> None:
        """POST /chat/completions/{completion_id}: missing mandatory 'metadata' input field → error."""

        def handler(body, **kwargs):
            return CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    f"/chat/completions/{COMPLETION_ID}", method="POST", body={}
                )
        finally:
            server.wait_for_completion()

    def test_update_incomplete_response_raises(self) -> None:
        """POST /chat/completions/{completion_id}: graph returns empty dict → mandatory output fields

        missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    f"/chat/completions/{COMPLETION_ID}",
                    method="POST",
                    body={"metadata": {"session": "abc"}},
                )
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # DELETE /chat/completions/{completion_id}
    # ---------------------------------------------------------------------------

    def test_delete_path_param_extracted_and_returns_correct_shape(self) -> None:
        """DELETE /chat/completions/{completion_id}: path param extracted; extra output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return DELETE_CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test(f"/chat/completions/{COMPLETION_ID}", method="DELETE")
            assert captured_kwargs.get("completion_id") == COMPLETION_ID
            assert "extra_field" not in resp
            assert resp["id"] == COMPLETION_ID
            assert resp["deleted"] is True
            assert resp["object"] == "chat.completion"
        finally:
            server.wait_for_completion()

    def test_delete_incomplete_response_raises(self) -> None:
        """DELETE /chat/completions/{completion_id}: graph returns empty dict → mandatory output fields missing → error."""  # noqa: E501

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/chat/completions/{COMPLETION_ID}", method="DELETE")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # GET /chat/completions
    # ---------------------------------------------------------------------------

    def test_list_filters_extra_output_fields(self) -> None:
        """GET /chat/completions: extra graph response fields filtered from output."""

        def handler(body, **kwargs):
            return LIST_CHAT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test("/chat/completions", method="GET")
            assert "extra_field" not in resp
            for key, value in LIST_CHAT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_list_incomplete_response_raises(self) -> None:
        """GET /chat/completions: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test("/chat/completions", method="GET")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # GET /chat/completions/{completion_id}/messages
    # ---------------------------------------------------------------------------

    def test_list_messages_path_param_extracted_and_output_filtered(self) -> None:
        """GET /chat/completions/{completion_id}/messages: path param extracted; extra output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return LIST_MESSAGES_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            resp = server.test(
                f"/chat/completions/{COMPLETION_ID}/messages", method="GET"
            )
            assert captured_kwargs.get("completion_id") == COMPLETION_ID
            assert "extra_field" not in resp
            for key, value in LIST_MESSAGES_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_list_messages_incomplete_response_raises(self) -> None:
        """GET /chat/completions/{completion_id}/messages: graph returns empty dict → mandatory output fields missing → error."""  # noqa: E501

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.CHAT_COMPLETIONS, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/chat/completions/{COMPLETION_ID}/messages", method="GET")
        finally:
            server.wait_for_completion()
