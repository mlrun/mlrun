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


import mlrun.common.types


class APIGatewayAuthenticationMode(mlrun.common.types.StrEnum):
    basic = "basicAuth"
    none = "none"
    access_key = "accessKey"
    iguazio = "iguazio"

    @classmethod
    def from_str(cls, authentication_mode: str):
        if authentication_mode == "none":
            return cls.none
        elif authentication_mode == "basicAuth":
            return cls.basic
        elif authentication_mode == "accessKey":
            return cls.access_key
        elif authentication_mode == "iguazio":
            return cls.iguazio
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Authentication mode `{authentication_mode}` is not supported",
            )


class APIGatewayState(mlrun.common.types.StrEnum):
    none = ""
    ready = "ready"
    error = "error"
    waiting_for_provisioning = "waitingForProvisioning"
