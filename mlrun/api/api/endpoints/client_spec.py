from fastapi import APIRouter, Depends

import mlrun.api.crud.client_spec
import mlrun.api.schemas
from mlrun.api.api import deps

router = APIRouter()


@router.get(
    "/client-spec", response_model=mlrun.api.schemas.ClientSpec,
)
def get_client_spec(auth_verifier: deps.AuthVerifier = Depends(deps.AuthVerifier),):
    return mlrun.api.crud.client_spec.ClientSpec().get_client_spec(auth_verifier)
