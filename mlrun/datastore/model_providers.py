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
from abc import ABC
from collections.abc import Awaitable
from typing import Callable, Optional, TypeVar

import mlrun
from mlrun.artifacts.llm_prompt import LLMPromptArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.datastore.abstract_base import (
    BaseRemoteClient,
    BaseRemoteClientManager,
    parse_url,
)
from mlrun.errors import err_to_str

from .store_resources import ResourceRemoteClient, get_store_resource

T = TypeVar("T")


class ModelProvider(BaseRemoteClient, ABC):
    support_async = False

    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs or {}
        self._client = None
        self._default_operation = None

    def load_client(self) -> None:
        raise NotImplementedError("load_client method is not implemented")

    @property
    def client(self):
        return self._client

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        return endpoint, subpath


class AsyncModelProvider(ModelProvider, ABC):
    support_async = True

    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            parent=parent,
            name=name,
            kind=kind,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self._async_client = None
        self._default_async_operation = None

    @property
    def async_client(self):
        return self._async_client


class OpenAIProvider(AsyncModelProvider):
    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.openai_default_model
        super().__init__(
            parent=parent,
            name=name,
            kind=kind,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self.options = self.get_client_options()
        self.load_client()

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        endpoint = endpoint + subpath
        #  in openai there is no usage of subpath variable. if the model contains "/", it is part of the model name.
        subpath = ""
        return endpoint, subpath

    @property
    def model(self):
        return self.endpoint

    def load_client(self) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI  # noqa

            self._client = OpenAI(**self.options)
            self._default_operation = self.client.ChatCompletion.create

            self._async_client = AsyncOpenAI(**self.options)
            self._default_async_operation = self.async_client.chat.completions.create
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc

    def get_client_options(self):
        res = dict(
            api_key=self._get_secret_or_env("OPENAI_API_KEY"),
            endpoint_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            open_ai_project_id=self._get_secret_or_env("OPENAI_ORG_ID"),
            openai_org_id=self._get_secret_or_env("OPENAI_PROJECT_ID"),
            timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        if operation:
            return operation(**kwargs, model=self.model)
        else:
            return self._default_operation(**invoke_kwargs, model=self.model)

    async def async_invoke(
        self,
        async_operation: Optional[Callable[..., Awaitable[T]]] = None,
        **invoke_kwargs,
    ) -> Awaitable[T]:
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        if async_operation:
            return async_operation(**kwargs, model=self.model)
        else:
            return self._default_async_operation(**invoke_kwargs, model=self.model)

    def basic_llm_invoke(self, prompt):
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        self._default_operation(model=self.endpoint, messages=messages)


def schema_to_model_provider(schema: str) -> type[ModelProvider]:
    #  TODO add hugging face and http
    schema_dict = {"openai": OpenAIProvider}
    provider_class = schema_dict.get(schema, None)
    if not provider_class:
        raise ValueError(f"unsupported model provider scheme ({schema})")
    return provider_class


class ModelProviderManager(BaseRemoteClientManager):
    def __init__(self, secrets=None, db=None):
        super().__init__(secrets=secrets, db=db)

    def get_or_create_model_provider(
        self, url, secrets: Optional[dict] = None, project_name="", default_invoke_kwargs: Optional[dict] = None
    ) -> (ModelProvider, str, str):
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path

        if schema == "ds":
            secrets, url, schema, endpoint, parsed_url, subpath = (
                self._resolve_datastore_profile(
                    url=url, secrets=secrets, project_name=project_name, subpath=subpath
                )
            )

        model_provider_class = schema_to_model_provider(schema)
        endpoint, subpath = model_provider_class.parse_endpoint_and_path(
            endpoint=endpoint, subpath=subpath
        )
        key = f"{schema}://{endpoint}" if endpoint else f"{schema}://"

        model_provider = model_provider_class(
            parent=self,
            name=key,
            kind=schema,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        return model_provider

    def get_model_artifact(
        self, url, project="", allow_empty_resources=None, secrets=None
    ):
        try:
            resource = get_store_resource(
                url,
                db=self._get_db(),
                secrets=self._secrets,
                project=project,
                data_store_secrets=secrets,
                fallback_manager=ResourceRemoteClient.MODEL_PROVIDER,
            )
        except Exception as exc:
            raise OSError(f"artifact {url} not found, {err_to_str(exc)}")
        if not isinstance(resource, ModelArtifact):
            raise mlrun.errors.MLRunRuntimeError(
                "The resource is not a ModelArtifact"
            )
        url = resource.model_url
        if not url and not allow_empty_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Resource {url} does not have model url"
            )
        return resource

    def object(
        self,
        url,
        key="",
        project="",
        allow_empty_resources=None,
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ) -> ModelProvider:
        if mlrun.datastore.is_store_uri(url):
            resource = self.get_model_artifact(
                url, project, allow_empty_resources, secrets
            )
            url = resource.model_url
            default_invoke_kwargs = default_invoke_kwargs or resource.default_config
        model_provider = self.get_or_create_model_provider(
            url, secrets=secrets, project_name=project, default_invoke_kwargs=default_invoke_kwargs
        )
        return model_provider
