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

"""Test fixtures for OpenAI endpoint body-mapping tests."""

# ---------------------------------------------------------------------------
# Shared: Response object
# Returned by POST /responses and GET /responses/{response_id}.
# ---------------------------------------------------------------------------

RESPONSE_OBJECT_HANDLER_RESPONSE = {
    "id": "resp_123",
    "created_at": 1234567890,
    "error": None,
    "incomplete_details": None,
    "instructions": "Be helpful",
    "metadata": {},
    "model": "gpt-4",
    "object": "response",
    "output": [{"type": "message", "role": "assistant", "content": []}],
    "parallel_tool_calls": True,
    "temperature": 1.0,
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "status": "completed",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "extra_field": "should_be_filtered",
}

RESPONSE_OBJECT_EXPECTED_RESPONSE = {
    k: v for k, v in RESPONSE_OBJECT_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# POST /responses
# ---------------------------------------------------------------------------

CREATE_REQUEST_BODY = {
    "background": False,
    "context_management": {"type": "auto"},
    "conversation": "conv_1",
    "include": ["reasoning.encrypted_content"],
    "input": "Hello",
    "instructions": "Be helpful",
    "max_output_tokens": 1024,
    "max_tool_calls": 5,
    "metadata": {"key": "value"},
    "model": "gpt-4",
    "parallel_tool_calls": True,
    "previous_response_id": "resp_0",
    "prompt": {"id": "prompt_1"},
    "prompt_cache_key": "cache_key_1",
    "prompt_cache_retention": True,
    "reasoning": {"effort": "medium"},
    "safety_identifier": "user_123",
    "service_tier": "default",
    "store": True,
    "stream": False,
    "stream_options": {"include_usage": True},
    "temperature": 0.7,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [{"type": "web_search"}],
    "top_logprobs": 2,
    "top_p": 0.9,
    "truncation": "disabled",
    "extra_field": "should_be_filtered",
}

# Same as above minus extra_field — what the handler should receive.
CREATE_EXPECTED_KWARGS = {
    k: v for k, v in CREATE_REQUEST_BODY.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# GET /responses/{response_id}
# ---------------------------------------------------------------------------

GET_RESPONSE_ID = "resp_123"

# ---------------------------------------------------------------------------
# GET /responses/{response_id}/input_items
# ---------------------------------------------------------------------------

INPUT_ITEMS_RESPONSE_ID = "resp_123"

INPUT_ITEMS_HANDLER_RESPONSE = {
    "data": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ],
    "first_id": "item_001",
    "has_more": False,
    "last_id": "item_001",
    "object": "list",
    "extra_field": "should_be_filtered",
}

INPUT_ITEMS_EXPECTED_RESPONSE = {
    k: v for k, v in INPUT_ITEMS_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# DELETE /responses/{response_id}
# ---------------------------------------------------------------------------

DELETE_RESPONSE_ID = "resp_123"

DELETE_HANDLER_RESPONSE = {
    "id": DELETE_RESPONSE_ID,
    "object": "response",
    "deleted": True,
}

# ---------------------------------------------------------------------------
# POST /responses/compact
# ---------------------------------------------------------------------------

COMPACT_REQUEST_BODY = {
    "model": "gpt-4",
    "input": "Hello",
    "instructions": "Be helpful",
    "previous_response_id": "resp_0",
    "prompt_cache_key": "cache_key_1",
    "prompt_cache_retention": True,
    "service_tier": "default",
    "extra_field": "should_be_filtered",
}

COMPACT_EXPECTED_KWARGS = {
    k: v for k, v in COMPACT_REQUEST_BODY.items() if k != "extra_field"
}

COMPACT_HANDLER_RESPONSE = {
    "id": "resp_123",
    "object": "response.compaction",
    "created_at": 1234567890,
    "output": [{"type": "text", "text": "Hello"}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "extra_field": "should_be_filtered",
}

COMPACT_EXPECTED_RESPONSE = {
    k: v for k, v in COMPACT_HANDLER_RESPONSE.items() if k != "extra_field"
}
