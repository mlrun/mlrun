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
from urllib.parse import urlparse

import mlrun


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

    @staticmethod
    def _sanitize_options(options):
        if not options:
            return {}
        options = {k: v for k, v in options.items() if v is not None and v != ""}
        return options
