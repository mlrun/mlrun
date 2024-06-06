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

from fastapi import APIRouter, Depends, Header

import mlrun.common.schemas
import server.api.api.utils
import server.api.crud
import server.api.utils.helpers

router = APIRouter()


@server.api.utils.helpers.lru_cache_with_ttl(maxsize=32, ttl_seconds=60 * 5)
def get_cached_client_spec(
    client_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    client_python_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.python_version
    ),
) -> mlrun.common.schemas.ClientSpec:
    return server.api.crud.ClientSpec().get_client_spec(
        client_version,
        client_python_version,
    )


@router.get(
    "/client-spec",
    response_model=mlrun.common.schemas.ClientSpec,
)
def get_client_spec(
    client_spec: mlrun.common.schemas.ClientSpec = Depends(get_cached_client_spec),
):
    return client_spec
