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
from typing import Any, Optional
from urllib.parse import urlparse

from mergedeep import merge

import mlrun
from mlrun.datastore.datastore_profile import datastore_profile_read


def parse_url(url):
    if url and url.startswith("v3io://") and not url.startswith("v3io:///"):
        url = url.replace("v3io://", "v3io:///", 1)
    parsed_url = urlparse(url)
    schema = parsed_url.scheme.lower()
    endpoint = parsed_url.hostname
    if endpoint:
        # HACK - urlparse returns the hostname after in lower case - we want the original case:
        # the hostname is a substring of the netloc, in which it's the original case, so we find the indexes of the
        # hostname in the netloc and take it from there
        lower_hostname = parsed_url.hostname
        netloc = str(parsed_url.netloc)
        lower_netloc = netloc.lower()
        hostname_index_in_netloc = lower_netloc.index(str(lower_hostname))
        endpoint = netloc[
            hostname_index_in_netloc : hostname_index_in_netloc + len(lower_hostname)
        ]
    if parsed_url.port:
        endpoint += f":{parsed_url.port}"
    return schema, endpoint, parsed_url


class BaseRemoteClient(ABC):
    def __init__(self, parent, name, kind, endpoint="", secrets: Optional[dict] = None):
        self._parent = parent
        self.name = name
        self.kind = kind
        self.endpoint = endpoint
        self._secrets = secrets or {}
        self.secret_pfx = ""  # TODO decide if needed here, or only in datastore.

    def _get_secret_or_env(self, key, default=None):
        # Project-secrets are mounted as env variables whose name can be retrieved from SecretsStore
        return mlrun.get_secret_or_env(
            key, secret_provider=self._get_secret, default=default
        )

    def _get_parent_secret(self, key):
        return self._parent.secret(self.secret_pfx + key)

    def _get_secret(self, key: str, default=None):
        return self._secrets.get(key, default) or self._get_parent_secret(key)

    @property
    def url(self):
        return f"{self.kind}://{self.endpoint}"

    @staticmethod
    def _sanitize_options(options):
        if not options:
            return {}
        options = {k: v for k, v in options.items() if v is not None and v != ""}
        return options


class BaseRemoteClientManager(ABC):
    def __init__(self, secrets=None, db=None):
        self._secrets = secrets or {}
        self._db = db

    @abstractmethod
    def object(
        self,
        url,
        key="",
        project="",
        allow_empty_resources=None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        pass

    @staticmethod
    def _resolve_datastore_profile(
        url,
        secrets: Optional[dict] = None,
        project_name="",
        subpath: Optional[str] = None,
    ):
        datastore_profile = datastore_profile_read(url, project_name, secrets)
        if secrets and datastore_profile.secrets():
            secrets = merge(secrets, datastore_profile.secrets())
        else:
            secrets = secrets or datastore_profile.secrets()
        url = datastore_profile.url(subpath)
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path
        return secrets, url, schema, endpoint, parsed_url, subpath

    def set(self, secrets=None, db=None):
        if db and not self._db:
            self._db = db
        if secrets:
            for key, val in secrets.items():
                self._secrets[key] = val
        return self

    def secret(self, key):
        return self._secrets.get(key)

    def _get_db(self):
        if not self._db:
            self._db = mlrun.get_run_db(secrets=self._secrets)
        return self._db

    def reset_secrets(self):
        self._secrets = {}
