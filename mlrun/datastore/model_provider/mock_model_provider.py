# Copyright 2023 Iguazio
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

from typing import Any, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
    UsageResponseKeys,
)


class MockModelProvider(ModelProvider):
    support_async = True
    supports_streaming = True

    def __init__(
        self,
        parent,
        kind,
        name,
        endpoint="",
        secrets: dict | None = None,
        default_invoke_kwargs: dict | None = None,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs or {}
        self._client = None
        self._async_client = None

    @staticmethod
    def _extract_string_output(response: Any) -> str:
        """
        Extracts string response from response object
        """
        pass

    def load_client(self) -> None:
        """
        Initializes the SDK client for the model provider with the given keyword arguments
        and assigns it to an instance attribute (e.g., self._client).

        Subclasses should override this method to:
        - Create and configure the provider-specific client instance.
        - Assign the client instance to self._client.
        """

        pass

    def _single_invoke(
        self,
        messages: list[dict],
        invoke_response_format: InvokeResponseFormat,
        counter: int | None = None,
    ) -> Union[dict[str, Any], str]:
        """
        Handle a single invocation. Raises error if message contains ERROR keyword.
        """
        text_response = (
            "You are using a mock model provider, no actual inference is performed."
        )
        # Add counter to text response if counter exists (including 0)
        if counter is not None:
            text_response = f"{text_response} (Item {counter})"

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Raise error if message contains "ERROR" keyword
        if any("ERROR" in msg.get("content", "") for msg in messages):
            raise RuntimeError("Mock error triggered by ERROR keyword in message")

        if invoke_response_format == InvokeResponseFormat.STRING:
            return text_response
        elif invoke_response_format == InvokeResponseFormat.FULL:
            return {
                UsageResponseKeys.USAGE: usage,
                UsageResponseKeys.ANSWER: text_response,
                "extra": {},
            }
        elif invoke_response_format == InvokeResponseFormat.USAGE:
            return {
                UsageResponseKeys.ANSWER: text_response,
                UsageResponseKeys.USAGE: usage,
            }
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported invoke response format: {invoke_response_format}"
            )

    def invoke(
        self,
        messages: Union[list[dict], list[list[dict]], Any],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, dict[str, Any], list[dict[str, Any]], Any]:
        is_batch = self._validate_and_detect_batch_invocation(messages)
        if is_batch:
            # Return list of mock responses with counter
            results = []
            for idx, msg_list in enumerate(messages):
                result = self._single_invoke(
                    msg_list, invoke_response_format, counter=idx
                )
                results.append(result)
            return results

        # Single invocation
        return self._single_invoke(messages, invoke_response_format)

    async def async_invoke(
        self,
        messages: Union[list[dict], list[list[dict]], Any],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, dict[str, Any], list[dict[str, Any]], Any]:
        return self.invoke(messages, invoke_response_format, **invoke_kwargs)

    def _stream_text(self, messages: list[dict]) -> str:
        """Generate the text response for streaming, with error checking."""
        if any("ERROR" in msg.get("content", "") for msg in messages):
            raise RuntimeError("Mock error triggered by ERROR keyword in message")
        return "You are using a mock model provider, no actual inference is performed."

    def invoke_stream(self, messages, **invoke_kwargs):
        if self._validate_and_detect_batch_invocation(messages):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Batch invocation is not supported in streaming mode"
            )
        text = self._stream_text(messages)
        for word in text.split():
            yield word + " "

    async def async_invoke_stream(self, messages, **invoke_kwargs):
        if self._validate_and_detect_batch_invocation(messages):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Batch invocation is not supported in streaming mode"
            )
        text = self._stream_text(messages)
        for word in text.split():
            yield word + " "
