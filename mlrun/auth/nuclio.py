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

import base64

import requests.auth
from nuclio.auth import AuthInfo as _NuclioAuthInfo
from nuclio.auth import AuthKinds as NuclioAuthKinds

import mlrun.auth.providers
import mlrun.common.schemas.auth


class NuclioAuthInfo(_NuclioAuthInfo):
    def __init__(self, token=None, **kwargs):
        super().__init__(**kwargs)
        self._token = token

    @classmethod
    def from_auth_info(cls, auth_info: "mlrun.common.schemas.auth.AuthInfo"):
        if not auth_info:
            return None
        if mlrun.mlconf.is_iguazio_v4_mode():
            return cls.from_request_headers(auth_info.request_headers)
        if auth_info.session != "":
            return NuclioAuthInfo(
                password=auth_info.session, mode=NuclioAuthKinds.iguazio
            )
        return None

    @classmethod
    def from_request_headers(cls, headers: dict[str, str]):
        if not headers:
            return cls()
        for key, value in headers.items():
            if key.lower() == "authorization":
                if value.lower().startswith("bearer "):
                    return cls(
                        token=value[len("bearer ") :],
                        mode=NuclioAuthKinds.iguazio,
                    )
                if value.lower().startswith("basic "):
                    token = value[len("basic ") :]
                    decoded_token = base64.b64decode(token).decode("utf-8")
                    username, password = decoded_token.split(":", 1)
                    return cls(
                        username=username,
                        password=password,
                        mode=NuclioAuthKinds.iguazio,
                    )
        return cls()

    @classmethod
    def from_envvar(cls):
        if mlrun.mlconf.is_iguazio_v4_mode():
            token_provider = mlrun.auth.providers.IGTokenProvider(
                token_endpoint=mlrun.mlconf.auth_token_endpoint,
            )
            return cls(
                token=token_provider.get_token(),
                mode=NuclioAuthKinds.iguazio,
            )
        return super().from_envvar()

    def to_requests_auth(self) -> "requests.auth":
        if self._token:
            # in iguazio v4 mode we use bearer token auth
            return _RequestAuthBearerToken(self._token)
        return super().to_requests_auth()


class _RequestAuthBearerToken(requests.auth.AuthBase):
    def __init__(self, token: str):
        self._token = token

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self._token}"
        return r
