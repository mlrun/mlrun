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
from typing import Optional

import mlrun


class BaseRemoteClient(ABC):
    def __init__(self, parent, name, kind, endpoint="", secrets: Optional[dict] = None):
        self._parent = parent
        self.name = name
        self.kind = kind
        self.endpoint = endpoint
        self._secrets = secrets or {}
        self.secret_pfx = ""

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
