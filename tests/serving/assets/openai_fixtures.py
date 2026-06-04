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

# Shared response_id used by all endpoints that take a {response_id} path parameter.
RESPONSE_ID = "resp_123"

# ---------------------------------------------------------------------------
# Shared: Response object
# Returned by POST /responses, GET /responses/{response_id},
# and POST /responses/{response_id}/cancel.
# ---------------------------------------------------------------------------

RESPONSE_OBJECT_HANDLER_RESPONSE = {
    "id": RESPONSE_ID,
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
    "store": True,
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
# GET /responses/{response_id}/input_items
# ---------------------------------------------------------------------------

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
# POST /responses/input_tokens
# ---------------------------------------------------------------------------

INPUT_TOKENS_REQUEST_BODY = {
    "conversation": "conv_1",
    "input": "Hello",
    "instructions": "Be helpful",
    "model": "gpt-4",
    "parallel_tool_calls": True,
    "previous_response_id": "resp_0",
    "reasoning": {"effort": "medium"},
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [{"type": "web_search"}],
    "truncation": "disabled",
    "extra_field": "should_be_filtered",
}

INPUT_TOKENS_EXPECTED_KWARGS = {
    k: v for k, v in INPUT_TOKENS_REQUEST_BODY.items() if k != "extra_field"
}

INPUT_TOKENS_HANDLER_RESPONSE = {
    "input_tokens": 42,
    "object": "response.input_tokens",
    "extra_field": "should_be_filtered",
}

INPUT_TOKENS_EXPECTED_RESPONSE = {
    k: v for k, v in INPUT_TOKENS_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# DELETE /responses/{response_id}
# ---------------------------------------------------------------------------

DELETE_HANDLER_RESPONSE = {
    "id": RESPONSE_ID,
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
    "id": RESPONSE_ID,
    "object": "response.compaction",
    "created_at": 1234567890,
    "output": [{"type": "text", "text": "Hello"}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "extra_field": "should_be_filtered",
}

COMPACT_EXPECTED_RESPONSE = {
    k: v for k, v in COMPACT_HANDLER_RESPONSE.items() if k != "extra_field"
}

# Shared completion_id used by all endpoints that take a {completion_id} path parameter.
COMPLETION_ID = "chatcmpl_123"

# ---------------------------------------------------------------------------
# GET /chat/completions
# ---------------------------------------------------------------------------

LIST_CHAT_HANDLER_RESPONSE = {
    "data": [{"id": COMPLETION_ID, "object": "chat.completion"}],
    "first_id": COMPLETION_ID,
    "has_more": False,
    "last_id": COMPLETION_ID,
    "object": "list",
    "extra_field": "should_be_filtered",
}

LIST_CHAT_EXPECTED_RESPONSE = {
    k: v for k, v in LIST_CHAT_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# GET /chat/completions/{completion_id}/messages
# ---------------------------------------------------------------------------

LIST_MESSAGES_HANDLER_RESPONSE = {
    "data": [
        {"id": "msg_001", "content_parts": [{"type": "text", "text": "Hello"}]},
        {"id": "msg_002", "content_parts": [{"type": "text", "text": "World"}]},
    ],
    "first_id": "msg_001",
    "has_more": False,
    "last_id": "msg_002",
    "object": "list",
    "extra_field": "should_be_filtered",
}

LIST_MESSAGES_EXPECTED_RESPONSE = {
    k: v for k, v in LIST_MESSAGES_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# POST /chat/completions
# ---------------------------------------------------------------------------

CHAT_REQUEST_BODY = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "gpt-4",
    "audio": {"voice": "alloy", "format": "mp3"},
    "frequency_penalty": 0.5,
    "logit_bias": {"50256": -100},
    "logprobs": True,
    "max_completion_tokens": 512,
    "metadata": {"session": "abc"},
    "modalities": ["text"],
    "n": 1,
    "parallel_tool_calls": True,
    "prediction": {"type": "content", "content": "Hello"},
    "presence_penalty": 0.0,
    "prompt_cache_key": "chat_cache_1",
    "prompt_cache_retention": False,
    "reasoning_effort": "medium",
    "response_format": {"type": "text"},
    "safety_identifier": "user_456",
    "service_tier": "default",
    "stop": ["\n"],
    "store": False,
    "stream": False,
    "stream_options": {"include_usage": True},
    "temperature": 0.8,
    "tool_choice": "auto",
    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    "top_logprobs": 3,
    "top_p": 0.95,
    "verbosity": "verbose",
    "web_search_options": {"search_context_size": "medium"},
    "extra_field": "should_be_filtered",
}

CHAT_EXPECTED_KWARGS = {
    k: v for k, v in CHAT_REQUEST_BODY.items() if k != "extra_field"
}

CHAT_HANDLER_RESPONSE = {
    "id": COMPLETION_ID,
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ],
    "created": 1234567890,
    "model": "gpt-4",
    "object": "chat.completion",
    "service_tier": "default",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "extra_field": "should_be_filtered",
}

CHAT_EXPECTED_RESPONSE = {
    k: v for k, v in CHAT_HANDLER_RESPONSE.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# POST /chat/completions/{completion_id}
# ---------------------------------------------------------------------------

UPDATE_CHAT_REQUEST_BODY = {
    "metadata": {"session": "updated"},
    "extra_field": "should_be_filtered",
}

UPDATE_CHAT_EXPECTED_KWARGS = {
    k: v for k, v in UPDATE_CHAT_REQUEST_BODY.items() if k != "extra_field"
}

# ---------------------------------------------------------------------------
# DELETE /chat/completions/{completion_id}
# ---------------------------------------------------------------------------

DELETE_CHAT_HANDLER_RESPONSE = {
    "id": COMPLETION_ID,
    "deleted": True,
    "object": "chat.completion",
    "extra_field": "should_be_filtered",
}
