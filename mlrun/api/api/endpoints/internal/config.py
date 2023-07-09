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
import http

import fastapi

import mlrun.config

router = fastapi.APIRouter()


@router.put(
    "/expose-endpoints",
    responses={http.HTTPStatus.ACCEPTED.value: {}},
)
def expose_internal_api_endpoints():
    mlrun.config.config.debug.expose_internal_api_endpoints = True


@router.put(
    "/mask-endpoints",
    responses={http.HTTPStatus.ACCEPTED.value: {}},
)
def mask_internal_api_endpoints():
    mlrun.config.config.debug.expose_internal_api_endpoints = False
