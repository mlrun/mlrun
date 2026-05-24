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
