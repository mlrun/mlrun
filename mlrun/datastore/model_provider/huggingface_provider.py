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

from typing import Callable, Optional, TypeVar, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import ModelProvider

T = TypeVar("T")
ChatType = list[dict[str, str]]  # according to transformers.pipelines.text_generation


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFaceProvider is a wrapper around the OpenAI SDK that provides an interface
    for interacting with OpenAI's generative AI services.

    It supports both synchronous and asynchronous operations, allowing flexible
    integration into various workflows.

    This class extends the ModelProvider base class and implements OpenAI-specific
    functionality, including client initialization, model invocation, and custom
    operations tailored to the OpenAI API.
    """

    def __init__(
        self,
        parent,
        schema,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.huggingface_default_model
        if schema != "huggingface":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "HuggingFaceProvider supports only 'huggingface' as the provider kind."
            )
        super().__init__(
            parent=parent,
            kind=schema,
            name=name,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self.options = self.get_client_options()
        self.load_client()

    @staticmethod
    def _extract_string_output(result) -> str:
        """
        Extracts the first generated string from Hugging Face pipeline output,
        regardless of whether it's plain text-generation or chat-style output.
        """
        if not isinstance(result, list) or len(result) == 0:
            raise ValueError("Empty or invalid pipeline output")

        return result[0].get("generated_text")

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        if endpoint and subpath:
            endpoint = endpoint + subpath
            #  in hf there is no usage of subpath variable. if the model contains "/", it is part of the model name.
            subpath = ""
        return endpoint, subpath

    def load_client(self) -> None:
        try:
            from transformers import pipeline, AutoModelForCausalLM  # noqa
            from transformers import AutoTokenizer  # noqa

            self._client = pipeline(model=self.model, **self.options)
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

    def get_client_options(self):
        res = dict(
            task=self._get_secret_or_env("HF_TASK") or "text-generation",
            token=self._get_secret_or_env("HF_TOKEN"),
            device=self._get_secret_or_env("HF_DEVICE"),
            device_map=self._get_secret_or_env("HF_DEVICE_MAP"),
            trust_remote_code=self._get_secret_or_env("HF_TRUST_REMOTE_CODE"),
            model_kwargs=self._get_secret_or_env("HF_MODEL_KWARGS"),
        )
        return self._sanitize_options(res)

    def custom_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        """
        OpenAI-specific implementation of `ModelProvider.custom_invoke`.

        Invokes an OpenAI model operation using the sync client. For full details, see
        `ModelProvider.custom_invoke`.

        Example:
            ```python
            result = openai_model_provider.invoke(
                openai_model_provider.client.images.generate,
                prompt="A futuristic cityscape at sunset",
                n=1,
                size="1024x1024",
            )
            ```
        :param operation: A callable representing the model operation (e.g., a client method).
        :param invoke_kwargs: Keyword arguments to pass to the operation.
        :return: The full response returned by the operation.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            return operation(**invoke_kwargs)
        else:
            return self.client(**invoke_kwargs)

    def invoke(
        self,
        messages: Union[str, list[str], ChatType, list[ChatType]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> Optional[Union[str, list, T]]:
        if self.client.task != "text-generation":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "HuggingFaceProvider.invoke supports text-generation task only"
            )
        if as_str:
            invoke_kwargs["return_full_text"] = False
        response = self.custom_invoke(text_inputs=messages, **invoke_kwargs)
        if as_str:
            return self._extract_string_output(response)
        return response
