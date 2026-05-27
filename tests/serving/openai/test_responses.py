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

"""Mock-server tests for the Responses endpoint group."""

import pytest

from mlrun.serving.openai_mappings import OpenAIEndpoint
from tests.serving.assets.openai_fixtures import (
    COMPACT_EXPECTED_KWARGS,
    COMPACT_EXPECTED_RESPONSE,
    COMPACT_HANDLER_RESPONSE,
    COMPACT_REQUEST_BODY,
    CREATE_EXPECTED_KWARGS,
    CREATE_REQUEST_BODY,
    DELETE_HANDLER_RESPONSE,
    INPUT_ITEMS_EXPECTED_RESPONSE,
    INPUT_ITEMS_HANDLER_RESPONSE,
    INPUT_TOKENS_EXPECTED_KWARGS,
    INPUT_TOKENS_EXPECTED_RESPONSE,
    INPUT_TOKENS_HANDLER_RESPONSE,
    INPUT_TOKENS_REQUEST_BODY,
    RESPONSE_ID,
    RESPONSE_OBJECT_EXPECTED_RESPONSE,
    RESPONSE_OBJECT_HANDLER_RESPONSE,
)
from tests.serving.openai.openai_common import make_mock_server


class TestResponsesGroupMock:
    """End-to-end mock-server tests for the Responses endpoint group."""

    # ---------------------------------------------------------------------------
    # POST /responses
    # ---------------------------------------------------------------------------

    def test_create_filters_extra_input_and_output_fields(self) -> None:
        """POST /responses: extra request fields filtered from input;
        extra graph response fields filtered from output."""
        captured: dict = {}

        def handler(body, **kwargs):
            captured.update(kwargs)
            return RESPONSE_OBJECT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test("/responses", method="POST", body=CREATE_REQUEST_BODY)
            assert "extra_field" not in captured
            for key, value in CREATE_EXPECTED_KWARGS.items():
                assert captured[key] == value, f"kwargs[{key!r}] mismatch"
            assert "extra_field" not in resp
            for key, value in RESPONSE_OBJECT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_create_incomplete_response_raises(self) -> None:
        """POST /responses: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test("/responses", method="POST", body={})
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # GET /responses/{response_id}
    # ---------------------------------------------------------------------------

    def test_get_path_param_extracted_and_returns_response_object(self) -> None:
        """GET /responses/{response_id}: path param extracted; response matches Response object spec."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return RESPONSE_OBJECT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(f"/responses/{RESPONSE_ID}", method="GET")
            assert captured_kwargs.get("response_id") == RESPONSE_ID
            assert "extra_field" not in resp
            for key, value in RESPONSE_OBJECT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_get_incomplete_response_raises(self) -> None:
        """GET /responses/{response_id}: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/responses/{RESPONSE_ID}", method="GET")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # DELETE /responses/{response_id}
    # ---------------------------------------------------------------------------

    def test_delete_path_param_extracted_and_returns_correct_shape(self) -> None:
        """DELETE /responses/{response_id}: path param extracted; response matches spec."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return DELETE_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(f"/responses/{RESPONSE_ID}", method="DELETE")
            assert captured_kwargs.get("response_id") == RESPONSE_ID
            assert resp["id"] == RESPONSE_ID
            assert resp["deleted"] is True
            assert resp["object"] == "response"
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # GET /responses/{response_id}/input_items
    # ---------------------------------------------------------------------------

    def test_input_items_path_param_extracted_and_output_filtered(self) -> None:
        """GET /responses/{response_id}/input_items: path param extracted; extra output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return INPUT_ITEMS_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(f"/responses/{RESPONSE_ID}/input_items", method="GET")
            assert captured_kwargs.get("response_id") == RESPONSE_ID
            assert "extra_field" not in resp
            for key, value in INPUT_ITEMS_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_input_items_incomplete_response_raises(self) -> None:
        """GET /responses/{response_id}/input_items: graph returns empty dict → mandatory fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/responses/{RESPONSE_ID}/input_items", method="GET")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # POST /responses/input_tokens
    # ---------------------------------------------------------------------------

    def test_input_tokens_filters_extra_input_and_output_fields(self) -> None:
        """POST /responses/input_tokens: extra request fields filtered from input;
        extra graph response fields filtered from output."""
        captured: dict = {}

        def handler(body, **kwargs):
            captured.update(kwargs)
            return INPUT_TOKENS_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(
                "/responses/input_tokens",
                method="POST",
                body=INPUT_TOKENS_REQUEST_BODY,
            )
            assert "extra_field" not in captured
            for key, value in INPUT_TOKENS_EXPECTED_KWARGS.items():
                assert captured[key] == value, f"kwargs[{key!r}] mismatch"
            assert "extra_field" not in resp
            for key, value in INPUT_TOKENS_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_input_tokens_incomplete_response_raises(self) -> None:
        """POST /responses/input_tokens: graph returns empty dict → mandatory fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test("/responses/input_tokens", method="POST", body={})
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # POST /responses/{response_id}/cancel
    # ---------------------------------------------------------------------------

    def test_cancel_path_param_extracted_and_returns_response_object(self) -> None:
        """POST /responses/{response_id}/cancel: path param extracted; extra output fields filtered."""
        captured_kwargs: dict = {}

        def handler(body, **kwargs):
            captured_kwargs.update(kwargs)
            return RESPONSE_OBJECT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(f"/responses/{RESPONSE_ID}/cancel", method="POST")
            assert captured_kwargs.get("response_id") == RESPONSE_ID
            assert "extra_field" not in resp
            for key, value in RESPONSE_OBJECT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_cancel_incomplete_response_raises(self) -> None:
        """POST /responses/{response_id}/cancel: graph returns empty dict → mandatory fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(f"/responses/{RESPONSE_ID}/cancel", method="POST")
        finally:
            server.wait_for_completion()

    # ---------------------------------------------------------------------------
    # POST /responses/compact
    # ---------------------------------------------------------------------------

    def test_compact_filters_extra_input_and_output_fields(self) -> None:
        """POST /responses/compact: extra request fields filtered from input;
        extra graph response fields filtered from output."""
        captured: dict = {}

        def handler(body, **kwargs):
            captured.update(kwargs)
            return COMPACT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            resp = server.test(
                "/responses/compact", method="POST", body=COMPACT_REQUEST_BODY
            )
            assert "extra_field" not in captured
            for key, value in COMPACT_EXPECTED_KWARGS.items():
                assert captured[key] == value, f"kwargs[{key!r}] mismatch"
            assert "extra_field" not in resp
            for key, value in COMPACT_EXPECTED_RESPONSE.items():
                assert resp[key] == value, f"resp[{key!r}] mismatch"
        finally:
            server.wait_for_completion()

    def test_compact_incomplete_response_raises(self) -> None:
        """POST /responses/compact: graph returns empty dict → mandatory output fields missing → error."""

        def handler(body, **kwargs):
            return {}

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/responses/compact",
                    method="POST",
                    body={"model": "gpt-4"},
                )
        finally:
            server.wait_for_completion()

    def test_compact_missing_mandatory_model_raises(self) -> None:
        """POST /responses/compact: missing mandatory 'model' input field → error."""

        def handler(body, **kwargs):
            return COMPACT_HANDLER_RESPONSE

        server = make_mock_server(OpenAIEndpoint.RESPONSES, handler)
        try:
            with pytest.raises(RuntimeError, match="Mandatory field"):
                server.test(
                    "/responses/compact",
                    method="POST",
                    body={"input": "Hello"},  # model omitted
                )
        finally:
            server.wait_for_completion()
