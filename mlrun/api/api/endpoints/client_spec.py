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

from fastapi import APIRouter, Header

import mlrun.api.crud
import mlrun.common.schemas

router = APIRouter()


@router.get(
    "/client-spec",
    response_model=mlrun.common.schemas.ClientSpec,
)
def get_client_spec(
    client_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    client_python_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.python_version
    ),
):
    return mlrun.api.crud.ClientSpec().get_client_spec(
        client_version=client_version, client_python_version=client_python_version
    )
