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

"""Pre-built OpenAI body mappings and endpoint registry for set_openai_frontend()."""

import abc
from enum import StrEnum
from http import HTTPMethod

from mlrun.serving.endpoint_mapping import BodyMappings


class OpenAIEndpoint(StrEnum):
    """Supported OpenAI operation groups for set_openai_frontend()."""

    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"
    AUDIO = "audio"
    IMAGES = "images"


# ---------------------------------------------------------------------------
# Per-group endpoint classes
# ---------------------------------------------------------------------------


class _OpenAIEndpointGroup(abc.ABC):
    """Base class for OpenAI endpoint groups."""

    @classmethod
    @abc.abstractmethod
    def endpoints(cls) -> list[dict]:
        """Return endpoint kwargs dicts for this group.

        :return: List of dicts suitable for APIHandlerConfig.add_endpoint_handler().
        """


class ResponsesEndpoints(_OpenAIEndpointGroup):
    """OpenAI /responses operation group — endpoint definitions and body mappings."""

    @staticmethod
    def _create_input_bm() -> BodyMappings:
        bm = BodyMappings()
        bm.add_mapping("$.background", destination_path="background")
        bm.add_mapping("$.context_management", destination_path="context_management")
        bm.add_mapping("$.conversation", destination_path="conversation")
        bm.add_mapping("$.include", destination_path="include")
        bm.add_mapping("$.input", destination_path="input")
        bm.add_mapping("$.instructions", destination_path="instructions")
        bm.add_mapping("$.max_output_tokens", destination_path="max_output_tokens")
        bm.add_mapping("$.max_tool_calls", destination_path="max_tool_calls")
        bm.add_mapping("$.metadata", destination_path="metadata")
        bm.add_mapping("$.model", destination_path="model")
        bm.add_mapping("$.parallel_tool_calls", destination_path="parallel_tool_calls")
        bm.add_mapping(
            "$.previous_response_id", destination_path="previous_response_id"
        )
        bm.add_mapping("$.prompt", destination_path="prompt")
        bm.add_mapping("$.prompt_cache_key", destination_path="prompt_cache_key")
        bm.add_mapping(
            "$.prompt_cache_retention", destination_path="prompt_cache_retention"
        )
        bm.add_mapping("$.reasoning", destination_path="reasoning")
        bm.add_mapping("$.safety_identifier", destination_path="safety_identifier")
        bm.add_mapping("$.service_tier", destination_path="service_tier")
        bm.add_mapping("$.store", destination_path="store")
        bm.add_mapping("$.stream", destination_path="stream")
        bm.add_mapping("$.stream_options", destination_path="stream_options")
        bm.add_mapping("$.temperature", destination_path="temperature")
        bm.add_mapping("$.text", destination_path="text")
        bm.add_mapping("$.tool_choice", destination_path="tool_choice")
        bm.add_mapping("$.tools", destination_path="tools")
        bm.add_mapping("$.top_logprobs", destination_path="top_logprobs")
        bm.add_mapping("$.top_p", destination_path="top_p")
        bm.add_mapping("$.truncation", destination_path="truncation")
        return bm

    @staticmethod
    def _create_output_bm() -> BodyMappings:
        bm = BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.created_at", destination_path="created_at", mandatory=True)
        bm.add_mapping("$.error", destination_path="error", mandatory=True)
        bm.add_mapping(
            "$.incomplete_details",
            destination_path="incomplete_details",
            mandatory=True,
        )
        bm.add_mapping(
            "$.instructions", destination_path="instructions", mandatory=True
        )
        bm.add_mapping("$.metadata", destination_path="metadata", mandatory=True)
        bm.add_mapping("$.model", destination_path="model", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        bm.add_mapping("$.output", destination_path="output", mandatory=True)
        bm.add_mapping(
            "$.parallel_tool_calls",
            destination_path="parallel_tool_calls",
            mandatory=True,
        )
        bm.add_mapping("$.temperature", destination_path="temperature", mandatory=True)
        bm.add_mapping("$.tool_choice", destination_path="tool_choice", mandatory=True)
        bm.add_mapping("$.tools", destination_path="tools", mandatory=True)
        bm.add_mapping("$.top_p", destination_path="top_p", mandatory=True)
        bm.add_mapping("$.background", destination_path="background")
        bm.add_mapping("$.completed_at", destination_path="completed_at")
        bm.add_mapping("$.conversation", destination_path="conversation")
        bm.add_mapping("$.max_output_tokens", destination_path="max_output_tokens")
        bm.add_mapping("$.max_tool_calls", destination_path="max_tool_calls")
        bm.add_mapping("$.output_text", destination_path="output_text")
        bm.add_mapping(
            "$.previous_response_id", destination_path="previous_response_id"
        )
        bm.add_mapping("$.prompt", destination_path="prompt")
        bm.add_mapping("$.prompt_cache_key", destination_path="prompt_cache_key")
        bm.add_mapping(
            "$.prompt_cache_retention", destination_path="prompt_cache_retention"
        )
        bm.add_mapping("$.reasoning", destination_path="reasoning")
        bm.add_mapping("$.safety_identifier", destination_path="safety_identifier")
        bm.add_mapping("$.service_tier", destination_path="service_tier")
        bm.add_mapping("$.status", destination_path="status")
        bm.add_mapping("$.text", destination_path="text")
        bm.add_mapping("$.top_logprobs", destination_path="top_logprobs")
        bm.add_mapping("$.truncation", destination_path="truncation")
        bm.add_mapping("$.usage", destination_path="usage")
        return bm

    @staticmethod
    def _compact_input_bm() -> BodyMappings:
        bm = BodyMappings()
        bm.add_mapping("$.model", destination_path="model", mandatory=True)
        bm.add_mapping("$.input", destination_path="input")
        bm.add_mapping("$.instructions", destination_path="instructions")
        bm.add_mapping(
            "$.previous_response_id", destination_path="previous_response_id"
        )
        bm.add_mapping("$.prompt_cache_key", destination_path="prompt_cache_key")
        bm.add_mapping(
            "$.prompt_cache_retention", destination_path="prompt_cache_retention"
        )
        bm.add_mapping("$.service_tier", destination_path="service_tier")
        return bm

    @staticmethod
    def _compact_output_bm() -> BodyMappings:
        bm = BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        bm.add_mapping("$.created_at", destination_path="created_at", mandatory=True)
        bm.add_mapping("$.output", destination_path="output", mandatory=True)
        bm.add_mapping("$.usage", destination_path="usage", mandatory=True)
        return bm

    @classmethod
    def endpoints(cls) -> list[dict]:
        return [
            {
                "path": "/responses",
                "http_method": HTTPMethod.POST,
                "input_body_mappings": cls._create_input_bm(),
                "output_body_mappings": cls._create_output_bm(),
            },
            {
                "path": "/responses/{response_id}",
                "http_method": HTTPMethod.DELETE,
            },
            {
                "path": "/responses/compact",
                "http_method": HTTPMethod.POST,
                "input_body_mappings": cls._compact_input_bm(),
                "output_body_mappings": cls._compact_output_bm(),
            },
        ]


class ChatCompletionsEndpoints(_OpenAIEndpointGroup):
    """OpenAI /chat/completions operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[dict]:
        return []  # TODO: ML-12461


class AudioEndpoints(_OpenAIEndpointGroup):
    """OpenAI /audio operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[dict]:
        return []  # TODO: ML-12461


class ImagesEndpoints(_OpenAIEndpointGroup):
    """OpenAI /images operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[dict]:
        return []  # TODO: ML-12461


# ---------------------------------------------------------------------------
# Internal mapping: OpenAIEndpoint → its group class
# ---------------------------------------------------------------------------

ENDPOINT_CLASSES: dict[OpenAIEndpoint, type[_OpenAIEndpointGroup]] = {
    OpenAIEndpoint.RESPONSES: ResponsesEndpoints,
    OpenAIEndpoint.CHAT_COMPLETIONS: ChatCompletionsEndpoints,
    OpenAIEndpoint.AUDIO: AudioEndpoints,
    OpenAIEndpoint.IMAGES: ImagesEndpoints,
}
