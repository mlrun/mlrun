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
import dataclasses
from enum import StrEnum
from http import HTTPMethod

from mlrun.serving.endpoint_mapping import BodyMappings


@dataclasses.dataclass
class OpenAIEndpointDef:
    """Definition of a single OpenAI endpoint."""

    path: str
    http_method: HTTPMethod
    description: str = ""
    input_body_mappings: BodyMappings | None = None
    output_body_mappings: BodyMappings | None = None

    def to_handler_kwargs(self) -> dict:
        """Return kwargs suitable for APIHandlerConfig.add_endpoint_handler()."""
        kwargs = {"path": self.path, "http_method": self.http_method}
        if self.input_body_mappings is not None:
            kwargs["input_body_mappings"] = self.input_body_mappings
        if self.output_body_mappings is not None:
            kwargs["output_body_mappings"] = self.output_body_mappings
        return kwargs


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
    def endpoints(cls) -> list[OpenAIEndpointDef]:
        """Return endpoint definitions for this group.

        :return: List of OpenAIEndpointDef instances.
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
        return bm

    @staticmethod
    def _compact_output_bm() -> BodyMappings:
        bm = BodyMappings()
        bm.add_mapping("$.id", destination_path="id")
        bm.add_mapping("$.object", destination_path="object")
        bm.add_mapping("$.created_at", destination_path="created_at")
        bm.add_mapping("$.model", destination_path="model")
        bm.add_mapping("$.status", destination_path="status")
        bm.add_mapping("$.output", destination_path="output")
        bm.add_mapping("$.usage", destination_path="usage")
        return bm

    @classmethod
    def endpoints(cls) -> list[OpenAIEndpointDef]:
        return [
            OpenAIEndpointDef(
                description="Delete a stored response by ID.",
                path="/responses/{response_id}",
                http_method=HTTPMethod.DELETE,
            ),
            OpenAIEndpointDef(
                description="Create a compacted response from prior context.",
                path="/responses/compact",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._compact_input_bm(),
                output_body_mappings=cls._compact_output_bm(),
            ),
        ]


class ChatCompletionsEndpoints(_OpenAIEndpointGroup):
    """OpenAI /chat/completions operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[OpenAIEndpointDef]:
        return []  # TODO: ML-12461


class AudioEndpoints(_OpenAIEndpointGroup):
    """OpenAI /audio operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[OpenAIEndpointDef]:
        return []  # TODO: ML-12461


class ImagesEndpoints(_OpenAIEndpointGroup):
    """OpenAI /images operation group — endpoint definitions and body mappings."""

    @classmethod
    def endpoints(cls) -> list[OpenAIEndpointDef]:
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
