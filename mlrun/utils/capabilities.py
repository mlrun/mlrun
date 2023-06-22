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
import typing

import mlrun
import mlrun.common.schemas.capabilities as capabilities


class Capabilities:
    @classmethod
    def k8s(cls):
        return cls._resolve_capability_mode(
            cls.k8s.__name__, mlrun.mlconf.is_api_running_on_k8s
        )

    @classmethod
    def kfp(cls):
        return cls._resolve_capability_mode(
            cls.kfp.__name__, mlrun.mlconf.resolve_kfp_url
        )

    @classmethod
    def iguazio(cls):
        return cls._resolve_capability_mode(
            cls.iguazio.__name__, mlrun.mlconf.get_parsed_igz_version
        )

    @classmethod
    def ce(cls):
        return cls._resolve_capability_mode(cls.ce.__name__, mlrun.mlconf.is_ce_mode)

    @classmethod
    def _resolve_capability_mode(
        cls, capability_name: str, is_capability_enabled_method: typing.Callable
    ):
        if cls._is_enabled(capability_name):
            return True
        elif is_capability_enabled_method():
            cls._update_capability_mode(capability_name)
            return True
        return False

    @staticmethod
    def _is_enabled(capability_name: str):
        return (
            getattr(mlrun.mlconf.capabilities, capability_name).mode
            == capabilities.CapabilitiesModes.enabled
        )

    @staticmethod
    def _update_capability_mode(
        capability_name: str,
        capability_mode: capabilities.CapabilitiesModes = capabilities.CapabilitiesModes.enabled,
    ):
        capability_config = getattr(mlrun.mlconf.capabilities, capability_name)
        capability_config.mode = capability_mode.value
