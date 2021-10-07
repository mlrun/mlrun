import fastapi

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier

router = fastapi.APIRouter()


@router.post("/authorization/verifications")
def verify_authorization(
    authorization_verification_input: mlrun.api.schemas.AuthorizationVerificationInput,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_permissions(
        authorization_verification_input.resource,
        authorization_verification_input.action,
        auth_info,
    )
