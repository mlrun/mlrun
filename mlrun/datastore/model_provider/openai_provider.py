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

from typing import Callable, Optional, TypeVar

import mlrun
from mlrun.datastore.model_provider.model_provider import ModelProvider

T = TypeVar("T")


class OpenAIProvider(ModelProvider):
    def __init__(
        self,
        parent,
        schema,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.openai_default_model
        if schema != "openai":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "OpenAIProvider supports only 'openai' as the provider kind."
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

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        if endpoint and subpath:
            endpoint = endpoint + subpath
            #  in openai there is no usage of subpath variable. if the model contains "/", it is part of the model name.
            subpath = ""
        return endpoint, subpath

    @property
    def model(self):
        return self.endpoint

    def load_client(self) -> None:
        try:
            from openai import OpenAI  # noqa

            self._client = OpenAI(**self.options)
            self._default_operation = self.client.chat.completions.create
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

    def get_client_options(self):
        res = dict(
            api_key=self._get_secret_or_env("OPENAI_API_KEY"),
            organization=self._get_secret_or_env("OPENAI_ORG_ID"),
            project=self._get_secret_or_env("OPENAI_PROJECT_ID"),
            base_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def customized_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            return operation(**invoke_kwargs, model=self.model)
        else:
            return self._default_operation(**invoke_kwargs, model=self.model)

    def _get_messages_parameter(
        self, prompt: Optional[str] = None, **invoke_kwargs
    ) -> (str, dict):
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        messages = invoke_kwargs.get("messages")
        if messages:
            if prompt:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "can not provide 'messages' and 'prompt' to invoke"
                )
        elif prompt:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "must provide 'messages' or 'prompt' to invoke"
            )
        return messages, invoke_kwargs

    def invoke(self, prompt: Optional[str] = None, **invoke_kwargs) -> str:
        messages, invoke_kwargs = self._get_messages_parameter(
            prompt=prompt, **invoke_kwargs
        )
        response = self._default_operation(
            model=self.endpoint, messages=messages, **invoke_kwargs
        )
        return response.choices[0].message.content
