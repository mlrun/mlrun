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
import fastapi

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier

router = fastapi.APIRouter()


@router.post("/authorization/verifications")
async def verify_authorization(
    authorization_verification_input: mlrun.api.schemas.AuthorizationVerificationInput,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_permissions(
        authorization_verification_input.resource,
        authorization_verification_input.action,
        auth_info,
    )
