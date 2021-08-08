import fastapi

import mlrun.api.api.deps
import mlrun.api.crud.client_spec
import mlrun.api.schemas

router = fastapi.APIRouter()


@router.get(
    "/client-spec", response_model=mlrun.api.schemas.ClientSpec,
)
def get_client_spec(
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    return mlrun.api.crud.client_spec.ClientSpec().get_client_spec()
