# Copyright 2018 Iguazio
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
from fastapi import APIRouter

import mlrun.api.crud
import mlrun.api.schemas

router = APIRouter()


@router.get(
    "/healthz",
    response_model=mlrun.api.schemas.ClientSpec,
)
def health():

    # TODO: From 0.7.0 client uses the /client-spec endpoint,
    #  when this is the oldest relevant client, remove this logic from the healthz endpoint
    return mlrun.api.crud.ClientSpec().get_client_spec()
