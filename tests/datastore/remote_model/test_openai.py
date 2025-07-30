# Copyright 2025 Iguazio
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

from mlrun.datastore.model_provider.openai_provider import OpenAIProvider

response_example = {
    "id": "1",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": None,
            "message": {
                "content": "Some generated message",
                "refusal": None,
                "role": "assistant",
                "annotations": [],
            },
        }
    ],
    "created": 1753784600,
    "model": "gpt-4o-mini-2024-07-18",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": "test",
    "usage": {
        "completion_tokens": 3,
        "prompt_tokens": 5,
        "total_tokens": 8,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
    },
}


def test_get_output_with_tokens_metrics():
    output, stats = OpenAIProvider.get_output_with_tokens_metrics(
        response=response_example
    )
    assert stats == {"completion_tokens": 3, "prompt_tokens": 5, "total_tokens": 8}
    assert output == "Some generated message"
