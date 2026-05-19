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
