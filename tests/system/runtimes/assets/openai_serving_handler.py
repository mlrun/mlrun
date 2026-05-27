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

"""Stub serving graph handlers for OpenAI frontend system tests."""

CHAT_COMPLETION_ID = "chatcmpl_system_test_123"
RESPONSE_ID = "resp_system_test_123"


def chat_completion_handler(body, **kwargs) -> dict:
    """Return a hard-coded ChatCompletion-shaped response.

    Includes extra_field to exercise output body mapping filtering.

    :param body: raw request body (unused)
    :param kwargs: mapped fields extracted by the input body mapping
    :return: ChatCompletion-shaped dict
    """
    return {
        "id": CHAT_COMPLETION_ID,
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"role": "assistant", "content": "Hello from MLRun!"},
            }
        ],
        "created": 1234567890,
        "model": kwargs.get("model", "gpt-4"),
        "object": "chat.completion",
        "service_tier": "default",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "extra_field": "should_be_filtered",
    }


def response_handler(body, **kwargs) -> dict:
    """Return a hard-coded Response-shaped response.

    Includes extra_field to exercise output body mapping filtering.

    :param body: raw request body (unused)
    :param kwargs: mapped fields extracted by the input body mapping
    :return: Response-shaped dict
    """
    return {
        "id": RESPONSE_ID,
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "completed_at": 1741476543,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": kwargs.get("model", "gpt-4"),
        "output": [
            {
                "type": "message",
                "id": "msg_system_test_001",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from MLRun!",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 36,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 87,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 123,
        },
        "metadata": {},
        "extra_field": "should_be_filtered",
    }


def chat_completion_handler_missing_mandatory(body, **kwargs) -> dict:
    """Return a ChatCompletion-shaped response with 'choices' omitted.

    :param body: raw request body (unused)
    :param kwargs: mapped fields extracted by the input body mapping
    :return: incomplete ChatCompletion dict — missing mandatory 'choices' output field
    """
    return {
        "id": CHAT_COMPLETION_ID,
        "created": 1234567890,
        "model": kwargs.get("model", "gpt-4"),
        "object": "chat.completion",
        # choices intentionally omitted — mandatory output field
    }


def response_handler_missing_mandatory(body, **kwargs) -> dict:
    """Return a Response-shaped response with 'id' omitted.

    :param body: raw request body (unused)
    :param kwargs: mapped fields extracted by the input body mapping
    :return: incomplete Response dict — missing mandatory 'id' output field
    """
    return {
        "object": "response",
        "created_at": 1741476542,
        "model": kwargs.get("model", "gpt-4"),
        # id intentionally omitted — mandatory output field
    }
