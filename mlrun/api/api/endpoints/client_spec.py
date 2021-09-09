from fastapi import APIRouter

import mlrun.api.crud
import mlrun.api.schemas

router = APIRouter()


@router.get(
    "/client-spec", response_model=mlrun.api.schemas.ClientSpec,
)
def get_client_spec():
    return mlrun.api.crud.ClientSpec().get_client_spec()
