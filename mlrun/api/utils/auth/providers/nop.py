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

import mlrun.api.utils.auth.providers.base
import mlrun.utils.singleton


class Provider(
    mlrun.api.utils.auth.providers.base.Provider,
    metaclass=mlrun.utils.singleton.AbstractSingleton,
):
    async def query_permissions(
        self,
        resource: str,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
        raise_on_forbidden: bool = True,
    ) -> bool:
        return True

    async def filter_by_permissions(
        self,
        resources: typing.List,
        opa_resource_extractor: typing.Callable,
        action: mlrun.common.schemas.AuthorizationAction,
        auth_info: mlrun.common.schemas.AuthInfo,
    ) -> typing.List:
        return resources

    def add_allowed_project_for_owner(
        self, project_name: str, auth_info: mlrun.common.schemas.AuthInfo
    ):
        pass
