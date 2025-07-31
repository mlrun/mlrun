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

from typing import TYPE_CHECKING, Any, Optional, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import ModelProvider

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline
    from transformers.pipelines.text_generation import ChatType


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFaceProvider is a wrapper around the Hugging Face Transformers pipeline
    that provides an interface for interacting with a wide range of Hugging Face models.

    It supports synchronous operations, enabling flexible integration into various workflows.

    This class extends the ModelProvider base class and implements Hugging Face-specific
    functionality, including pipeline initialization, default text generation operations,
    and custom operations tailored to the Hugging Face Transformers pipeline API.
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
        self._expected_operation_type = None
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
            # In HuggingFace, "/" in a model name is part of the name — `subpath` is not used.
            subpath = ""
        return endpoint, subpath

    def load_client(self) -> None:
        """
        Initializes the Hugging Face pipeline using the provided options.

        This method imports the `pipeline` function from the `transformers` package,
        creates a pipeline instance with the specified task and model (from `self.options`),
        and assigns it to `self._client`.

        Note: Hugging Face pipelines are synchronous and do not support async invocation.

        Raises:
            ImportError: If the `transformers` package is not installed.
        """
        try:
            from transformers import pipeline, AutoModelForCausalLM  # noqa
            from transformers import AutoTokenizer  # noqa
            from transformers.pipelines.base import Pipeline  # noqa

            self._client = pipeline(model=self.model, **self.options)
            self._expected_operation_type = Pipeline
        except ImportError as exc:
            raise ImportError("transformers package is not installed") from exc

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
        self, operation: Optional["Pipeline"] = None, **invoke_kwargs
    ) -> Union[list, dict, Any]:
        """
        HuggingFace implementation of `ModelProvider.custom_invoke`.
        Use the default config in provider client/ user defined client:

        Example:
        ```python
            image = Image.open(image_path)
            pipeline_object =  pipeline("image-classification", model="microsoft/resnet-50")
            result = hf_provider.custom_invoke(
                pipeline_object,
                inputs=image,
            )
        ```


        :param operation:               A pipeline object
        :param invoke_kwargs:           Keyword arguments to pass to the operation.
        :return:                        The full response returned by the operation.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            if not isinstance(operation, self._expected_operation_type):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Huggingface operation must inherit" " from 'Pipeline' object"
                )
            return operation(**invoke_kwargs)
        else:
            return self.client(**invoke_kwargs)

    def invoke(
        self,
        messages: Union[str, list[str], "ChatType", list["ChatType"]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> Union[str, list]:
        """
        HuggingFace-specific implementation of `ModelProvider.invoke`.
        Invokes a HuggingFace model operation using the synchronous client.
        For complete usage details, refer to `ModelProvider.invoke`.

        :param messages:
                            Same as ModelProvider.invoke.

        :param as_str:
                            If `True`, return only the main content (e.g., generated text) from a
                            **single-response output** — intended for use cases where you expect exactly one result.

                            If `False`, return the **full raw response object**, which is a list of dictionaries.

        :param invoke_kwargs:
                            Same as ModelProvider.invoke.
        :return:            Same as ModelProvider.invoke.
        """
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
