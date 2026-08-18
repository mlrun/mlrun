# Copyright 2026 Iguazio
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

import mlrun.common.types


class HTTPSessionRetryMode(mlrun.common.types.StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class HTTPTriggerAuthenticationMode(mlrun.common.types.StrEnum):
    """Authentication modes for a Nuclio function HTTP trigger (function-level, behind-Service auth).

    Set on the function's HTTP trigger ``attributes.authenticationMode`` and enforced by the auth
    sidecar injected into the function pod when the platform feature flag
    ``httpdb.nuclio.function_authentication_enabled`` is on. Different from
    :py:class:`~mlrun.common.schemas.APIGatewayAuthenticationMode`, which applies to API Gateways.
    """

    none = "none"
    api = "api"
    browser = "browser"
    basic = "basicAuth"

    @classmethod
    def values(cls) -> set[str]:
        """Return the set of supported authentication-mode string values."""
        return {m.value for m in cls}
