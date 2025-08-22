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

from typing import Any, Optional, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
    UsageResponseKeys,
)


class MockModelProvider(ModelProvider):
    support_async = False

    def __init__(
        self,
        parent,
        kind,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
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

    def invoke(
        self,
        messages: Union[list[dict], Any],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, dict[str, Any], Any]:
        if invoke_response_format == InvokeResponseFormat.STRING:
            return (
                "You are using a mock model provider, no actual inference is performed."
            )
        elif invoke_response_format == InvokeResponseFormat.FULL:
            return {
                UsageResponseKeys.USAGE: {"prompt_tokens": 0, "completion_tokens": 0},
                UsageResponseKeys.ANSWER: "You are using a mock model provider, no actual inference is performed.",
                "extra": {},
            }
        elif invoke_response_format == InvokeResponseFormat.USAGE:
            return {
                UsageResponseKeys.ANSWER: "You are using a mock model provider, no actual inference is performed.",
                UsageResponseKeys.USAGE: {"prompt_tokens": 0, "completion_tokens": 0},
            }
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unsupported invoke response format: {invoke_response_format}"
            )
