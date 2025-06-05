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
from abc import ABC, abstractmethod
from typing import Optional

from mlrun.datastore.abstract_base import BaseRemoteClient


class ModelProvider(BaseRemoteClient, ABC):
    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        **default_invoke_kwargs,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs

    @abstractmethod
    def get_client_options(self) -> dict:
        """retrieve provider secrets."""
        pass

    @abstractmethod
    def load_client(self) -> None:
        pass


class OpenAIProvider(ModelProvider):
    def __init__(
        self, parent, name, kind, endpoint="", secrets={}, **default_invoke_kwargs
    ):

        super().__init__(
            parent=parent,
            name=name,
            kind=kind,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self.client = None
        self._default_operation = None
        self.options = self.get_client_options()
        self.load_client()

    def _validate_model_name(self):
        # endpoint represent model name
        pass


    def load_client(self) -> None:
        try:
            from openai import OpenAI  # noqa

            self.client = OpenAI(**self.options)
            self._default_operation = self.client.ChatCompletion.create
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

    def invoke(self, operation: callable = None, **invoke_kwargs):
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        if operation:
            return operation(**kwargs, model=self.model)
        else:
            self._default_operation(**invoke_kwargs)
