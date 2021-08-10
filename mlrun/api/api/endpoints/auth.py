import fastapi

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.clients.opa

router = fastapi.APIRouter()


@router.post("/authorization/verifications")
def verify_authorization(
    authorization_verification_input: mlrun.api.schemas.AuthorizationVerificationInput,
    auth_verifier: mlrun.api.api.deps.AuthVerifierDep = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifierDep
    ),
):
    mlrun.api.utils.clients.opa.Client().query_permissions(
        authorization_verification_input.resource,
        authorization_verification_input.action,
        auth_verifier.auth_info,
    )
