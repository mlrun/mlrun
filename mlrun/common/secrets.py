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
#

from abc import ABC, abstractmethod

import mlrun.common.schemas


class SecretProviderInterface(ABC):
    @abstractmethod
    def store_auth_secret(
        self, username: str, access_key: str, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        pass

    @abstractmethod
    def delete_auth_secret(self, secret_ref: str, namespace=""):
        pass

    @abstractmethod
    def read_auth_secret(self, secret_name, namespace="", raise_on_not_found=False):
        pass

    @abstractmethod
    def store_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        pass

    @abstractmethod
    def delete_project_secrets(self, project, secrets, namespace=""):
        pass

    @abstractmethod
    def get_project_secret_keys(self, project, namespace="", filter_internal=False):
        pass

    @abstractmethod
    def get_project_secret_data(self, project, secret_keys=None, namespace=""):
        pass

    @abstractmethod
    def get_secret_data(self, secret_name, namespace=""):
        pass


class InMemorySecretProvider(SecretProviderInterface):
    def __init__(self):
        # project -> secret_key -> secret_value
        self.project_secrets_map = {}
        # ref -> secret_key -> secret_value
        self.auth_secrets_map = {}
        # secret-name -> secret_key -> secret_value
        self.secrets_map = {}

    def store_auth_secret(
        self, username: str, access_key: str, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        secret_ref = self.resolve_auth_secret_name(username, access_key)
        self.auth_secrets_map.setdefault(secret_ref, {}).update(
            self._generate_auth_secret_data(username, access_key)
        )
        return secret_ref, mlrun.common.schemas.SecretEventActions.created

    def delete_auth_secret(self, secret_ref: str, namespace=""):
        del self.auth_secrets_map[secret_ref]

    def read_auth_secret(self, secret_name, namespace="", raise_on_not_found=False):
        secret = self.auth_secrets_map.get(secret_name)
        if not secret:
            if raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Secret '{secret_name}' was not found in auth secrets map"
                )

            return None, None
        username = secret[
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("username")
        ]
        access_key = secret[
            mlrun.common.schemas.AuthSecretData.get_field_secret_key("access_key")
        ]
        return username, access_key

    def store_project_secrets(
        self, project, secrets, namespace=""
    ) -> (str, mlrun.common.schemas.SecretEventActions):
        self.project_secrets_map.setdefault(project, {}).update(secrets)
        secret_name = project
        return secret_name, mlrun.common.schemas.SecretEventActions.created

    def delete_project_secrets(self, project, secrets, namespace=""):
        if not secrets:
            self.project_secrets_map.pop(project, None)
        else:
            for key in secrets:
                self.project_secrets_map.get(project, {}).pop(key, None)
        return "", True

    def get_project_secret_keys(self, project, namespace="", filter_internal=False):
        secret_keys = list(self.project_secrets_map.get(project, {}).keys())
        if filter_internal:
            secret_keys = list(
                filter(lambda key: not key.startswith("mlrun."), secret_keys)
            )
        return secret_keys

    def get_project_secret_data(self, project, secret_keys=None, namespace=""):
        secrets_data = self.project_secrets_map.get(project, {})
        return {
            key: value
            for key, value in secrets_data.items()
            if (secret_keys and key in secret_keys) or not secret_keys
        }

    def store_secret(self, secret_name, secrets: dict):
        self.secrets_map[secret_name] = secrets

    def get_secret_data(self, secret_name, namespace=""):
        return self.secrets_map[secret_name]

    @staticmethod
    def _generate_auth_secret_data(username: str, access_key: str):
        return {
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "username"
            ): username,
            mlrun.common.schemas.AuthSecretData.get_field_secret_key(
                "access_key"
            ): access_key,
        }

    @staticmethod
    def resolve_auth_secret_name(username: str, access_key: str) -> str:
        return f"secret-ref-{username}-{access_key}"
