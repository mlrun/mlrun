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
import copy
from enum import StrEnum
from http import HTTPMethod

import mlrun.serving.endpoint_mapping as endpoint_mapping


class OpenAIEndpoint(StrEnum):
    """Supported OpenAI operation groups for set_openai_frontend()."""

    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


# ---------------------------------------------------------------------------
# Per-group endpoint classes
# ---------------------------------------------------------------------------


class _OpenAIEndpointGroup(abc.ABC):
    """Base class for OpenAI endpoint groups."""

    _ep_cache: list[endpoint_mapping.EndpointConfig] | None = None

    @classmethod
    def endpoints(cls) -> list[endpoint_mapping.EndpointConfig]:
        """Return endpoint configurations for this group (cached, deep-copied per call).

        :return: List of :class:`~mlrun.serving.endpoint_mapping.EndpointConfig` instances.
        """
        if cls._ep_cache is None:
            cls._ep_cache = cls._build_endpoints()
        return copy.deepcopy(cls._ep_cache)

    @classmethod
    @abc.abstractmethod
    def _build_endpoints(cls) -> list[endpoint_mapping.EndpointConfig]:
        """Build the endpoint configurations for this group.

        :return: List of :class:`~mlrun.serving.endpoint_mapping.EndpointConfig` instances.
        """


class ResponsesEndpoints(_OpenAIEndpointGroup):
    """OpenAI /responses operation group — endpoint definitions and body mappings."""

    @staticmethod
    def _create_input_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
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
    def _response_output_bm() -> endpoint_mapping.BodyMappings:
        """Shared output mapping for any endpoint returning a Response object."""
        bm = endpoint_mapping.BodyMappings()
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
        # not in the official OpenAI docs but present in the API response example
        bm.add_mapping("$.store", destination_path="store")
        return bm

    @staticmethod
    def _compact_input_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
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
    def _input_tokens_input_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.conversation", destination_path="conversation")
        bm.add_mapping("$.input", destination_path="input")
        bm.add_mapping("$.instructions", destination_path="instructions")
        bm.add_mapping("$.model", destination_path="model")
        bm.add_mapping("$.parallel_tool_calls", destination_path="parallel_tool_calls")
        bm.add_mapping(
            "$.previous_response_id", destination_path="previous_response_id"
        )
        bm.add_mapping("$.reasoning", destination_path="reasoning")
        bm.add_mapping("$.text", destination_path="text")
        bm.add_mapping("$.tool_choice", destination_path="tool_choice")
        bm.add_mapping("$.tools", destination_path="tools")
        bm.add_mapping("$.truncation", destination_path="truncation")
        return bm

    @staticmethod
    def _input_tokens_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping(
            "$.input_tokens", destination_path="input_tokens", mandatory=True
        )
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        return bm

    @staticmethod
    def _input_items_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.data", destination_path="data", mandatory=True)
        bm.add_mapping("$.first_id", destination_path="first_id", mandatory=True)
        bm.add_mapping("$.has_more", destination_path="has_more", mandatory=True)
        bm.add_mapping("$.last_id", destination_path="last_id", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        return bm

    @staticmethod
    def _compact_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        bm.add_mapping("$.created_at", destination_path="created_at", mandatory=True)
        bm.add_mapping("$.output", destination_path="output", mandatory=True)
        bm.add_mapping("$.usage", destination_path="usage", mandatory=True)
        return bm

    @classmethod
    def _build_endpoints(cls) -> list[endpoint_mapping.EndpointConfig]:
        return [
            endpoint_mapping.EndpointConfig(  # Create a response
                path="/responses",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._create_input_bm(),
                output_body_mappings=cls._response_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Retrieve a response
                path="/responses/{response_id}",
                http_method=HTTPMethod.GET,
                output_body_mappings=cls._response_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Delete a response — no output mapping: the return
                # shape ({id, object, deleted}) appears only in the API example, not in a formal
                # field table, so we treat it as passthrough rather than enforcing an undocumented
                # contract.
                path="/responses/{response_id}",
                http_method=HTTPMethod.DELETE,
            ),
            endpoint_mapping.EndpointConfig(  # List input items
                path="/responses/{response_id}/input_items",
                http_method=HTTPMethod.GET,
                output_body_mappings=cls._input_items_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Count input tokens
                path="/responses/input_tokens",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._input_tokens_input_bm(),
                output_body_mappings=cls._input_tokens_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Cancel a response
                path="/responses/{response_id}/cancel",
                http_method=HTTPMethod.POST,
                output_body_mappings=cls._response_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Compact a response
                path="/responses/compact",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._compact_input_bm(),
                output_body_mappings=cls._compact_output_bm(),
            ),
        ]


class ChatCompletionsEndpoints(_OpenAIEndpointGroup):
    """OpenAI /chat/completions operation group — endpoint definitions and body mappings."""

    @staticmethod
    def _chat_input_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.messages", destination_path="messages", mandatory=True)
        bm.add_mapping("$.model", destination_path="model", mandatory=True)
        bm.add_mapping("$.audio", destination_path="audio")
        bm.add_mapping("$.frequency_penalty", destination_path="frequency_penalty")
        bm.add_mapping("$.logit_bias", destination_path="logit_bias")
        bm.add_mapping("$.logprobs", destination_path="logprobs")
        bm.add_mapping(
            "$.max_completion_tokens", destination_path="max_completion_tokens"
        )
        bm.add_mapping("$.metadata", destination_path="metadata")
        bm.add_mapping("$.modalities", destination_path="modalities")
        bm.add_mapping("$.n", destination_path="n")
        bm.add_mapping("$.parallel_tool_calls", destination_path="parallel_tool_calls")
        bm.add_mapping("$.prediction", destination_path="prediction")
        bm.add_mapping("$.presence_penalty", destination_path="presence_penalty")
        bm.add_mapping("$.prompt_cache_key", destination_path="prompt_cache_key")
        bm.add_mapping(
            "$.prompt_cache_retention", destination_path="prompt_cache_retention"
        )
        bm.add_mapping("$.reasoning_effort", destination_path="reasoning_effort")
        bm.add_mapping("$.response_format", destination_path="response_format")
        bm.add_mapping("$.safety_identifier", destination_path="safety_identifier")
        bm.add_mapping("$.service_tier", destination_path="service_tier")
        bm.add_mapping("$.stop", destination_path="stop")
        bm.add_mapping("$.store", destination_path="store")
        bm.add_mapping("$.stream", destination_path="stream")
        bm.add_mapping("$.stream_options", destination_path="stream_options")
        bm.add_mapping("$.temperature", destination_path="temperature")
        bm.add_mapping("$.tool_choice", destination_path="tool_choice")
        bm.add_mapping("$.tools", destination_path="tools")
        bm.add_mapping("$.top_logprobs", destination_path="top_logprobs")
        bm.add_mapping("$.top_p", destination_path="top_p")
        bm.add_mapping("$.verbosity", destination_path="verbosity")
        bm.add_mapping("$.web_search_options", destination_path="web_search_options")
        return bm

    @staticmethod
    def _list_chat_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.data", destination_path="data", mandatory=True)
        bm.add_mapping("$.first_id", destination_path="first_id", mandatory=True)
        bm.add_mapping("$.has_more", destination_path="has_more", mandatory=True)
        bm.add_mapping("$.last_id", destination_path="last_id", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        return bm

    @staticmethod
    def _delete_chat_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.deleted", destination_path="deleted", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        return bm

    @staticmethod
    def _update_chat_input_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.metadata", destination_path="metadata", mandatory=True)
        return bm

    @staticmethod
    def _chat_completion_output_bm() -> endpoint_mapping.BodyMappings:
        bm = endpoint_mapping.BodyMappings()
        bm.add_mapping("$.id", destination_path="id", mandatory=True)
        bm.add_mapping("$.choices", destination_path="choices", mandatory=True)
        bm.add_mapping("$.created", destination_path="created", mandatory=True)
        bm.add_mapping("$.model", destination_path="model", mandatory=True)
        bm.add_mapping("$.object", destination_path="object", mandatory=True)
        bm.add_mapping("$.service_tier", destination_path="service_tier")
        bm.add_mapping("$.usage", destination_path="usage")
        return bm

    @classmethod
    def _build_endpoints(cls) -> list[endpoint_mapping.EndpointConfig]:
        return [
            endpoint_mapping.EndpointConfig(  # Create a chat completion
                path="/chat/completions",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._chat_input_bm(),
                output_body_mappings=cls._chat_completion_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Retrieve a chat completion
                path="/chat/completions/{completion_id}",
                http_method=HTTPMethod.GET,
                output_body_mappings=cls._chat_completion_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Update a chat completion
                path="/chat/completions/{completion_id}",
                http_method=HTTPMethod.POST,
                input_body_mappings=cls._update_chat_input_bm(),
                output_body_mappings=cls._chat_completion_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # Delete a chat completion
                path="/chat/completions/{completion_id}",
                http_method=HTTPMethod.DELETE,
                output_body_mappings=cls._delete_chat_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # List chat completions
                path="/chat/completions",
                http_method=HTTPMethod.GET,
                output_body_mappings=cls._list_chat_output_bm(),
            ),
            endpoint_mapping.EndpointConfig(  # List messages for a chat completion
                # Reuses _list_chat_output_bm() — top-level shape is identical;
                # nested ChatCompletionStoreMessage fields inside data are not checked.
                path="/chat/completions/{completion_id}/messages",
                http_method=HTTPMethod.GET,
                output_body_mappings=cls._list_chat_output_bm(),
            ),
        ]


# ---------------------------------------------------------------------------
# Internal mapping: OpenAIEndpoint → its group class
# ---------------------------------------------------------------------------

ENDPOINT_CLASSES: dict[OpenAIEndpoint, type[_OpenAIEndpointGroup]] = {
    OpenAIEndpoint.RESPONSES: ResponsesEndpoints,
    OpenAIEndpoint.CHAT_COMPLETIONS: ChatCompletionsEndpoints,
}
