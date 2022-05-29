# Copyright 2018 Iguazio
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

from ast import literal_eval
from os import environ

from .utils import AzureVaultStore, VaultStore, list2dict


class SecretsStore:
    def __init__(self):
        self._secrets = {}
        # Hidden secrets' value must not be serialized. Only the keys can be. These secrets are retrieved externally,
        # for example from Vault, and when adding their source they will be retrieved from the external source.
        self._hidden_sources = []
        self._hidden_secrets = {}
        self.vault = VaultStore()

    @classmethod
    def from_list(cls, src_list: list):
        store = cls()
        if src_list and isinstance(src_list, list):
            for src in src_list:
                store.add_source(src["kind"], src.get("source"), src.get("prefix", ""))
        return store

    def to_dict(self, struct):
        pass

    def add_source(self, kind, source="", prefix=""):

        if kind == "inline":
            if isinstance(source, str):
                source = literal_eval(source)
            if not isinstance(source, dict):
                raise ValueError("inline secrets must be of type dict")
            for k, v in source.items():
                self._secrets[prefix + k] = str(v)

        elif kind == "file":
            with open(source) as fp:
                lines = fp.read().splitlines()
                secrets_dict = list2dict(lines)
                for k, v in secrets_dict.items():
                    self._secrets[prefix + k] = str(v)

        elif kind == "env":
            for key in source.split(","):
                k = key.strip()
                self._secrets[prefix + k] = environ.get(k)

        elif kind == "vault":
            if isinstance(source, str):
                source = literal_eval(source)
            if not isinstance(source, dict):
                raise ValueError("vault secrets must be of type dict")

            for key, value in self.vault.get_secrets(
                source["secrets"],
                user=source.get("user"),
                project=source.get("project"),
            ).items():
                self._hidden_secrets[prefix + key] = value
            self._hidden_sources.append({"kind": kind, "source": source})

        elif kind == "azure_vault":
            if isinstance(source, str):
                source = literal_eval(source)
            if not isinstance(source, dict):
                raise ValueError("Azure vault secrets must be of type dict")
            if "name" not in source:
                raise ValueError(
                    "'name' must be provided in the source to define an Azure vault"
                )

            azure_vault = AzureVaultStore(source["name"])
            for key, value in azure_vault.get_secrets(source["secrets"]).items():
                self._hidden_secrets[prefix + key] = value
            self._hidden_sources.append({"kind": kind, "source": source})
        elif kind == "kubernetes":
            if isinstance(source, str):
                source = literal_eval(source)
            if not isinstance(source, list):
                raise ValueError("k8s secrets must be of type list")
            for secret in source:
                env_value = environ.get(self.k8s_env_variable_name_for_secret(secret))
                if env_value:
                    self._hidden_secrets[prefix + secret] = env_value
            self._hidden_sources.append({"kind": kind, "source": source})

    def get(self, key, default=None):
        return (
            self._secrets.get(key)
            or self._hidden_secrets.get(key)
            or environ.get(self.k8s_env_variable_name_for_secret(key))
            or default
        )

    def items(self):
        res = self._secrets.copy()
        if self._hidden_secrets:
            res.update(self._hidden_secrets)
        return res.items()

    def to_serial(self):
        # todo: use encryption
        res = [{"kind": "inline", "source": self._secrets.copy()}]
        if self._hidden_sources:
            for src in self._hidden_sources.copy():
                res.append(src)
        return res

    def has_vault_source(self):
        return any(source["kind"] == "vault" for source in self._hidden_sources)

    def has_azure_vault_source(self):
        return any(source["kind"] == "azure_vault" for source in self._hidden_sources)

    def get_azure_vault_k8s_secret(self):
        for source in self._hidden_sources:
            if source["kind"] == "azure_vault":
                return source["source"].get("k8s_secret", None)

    @staticmethod
    def k8s_env_variable_name_for_secret(secret_name):
        from mlrun.config import config

        return config.secret_stores.kubernetes.env_variable_prefix + secret_name.upper()

    def get_k8s_secrets(self):
        for source in self._hidden_sources:
            if source["kind"] == "kubernetes":
                return {
                    secret: self.k8s_env_variable_name_for_secret(secret)
                    for secret in source["source"]
                }
        return None
