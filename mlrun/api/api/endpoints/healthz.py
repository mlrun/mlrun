from fastapi import APIRouter

import mlrun.api.crud.client_spec
import mlrun.api.schemas

router = APIRouter()


# TODO: From 0.7.0 client uses the /client_spec endpoint,
#  when this is the oldest relevant client, remove this logic from the healthz endpoint
@router.get(
    "/healthz", response_model=mlrun.api.schemas.ClientSpec,
)
def health():
    return mlrun.api.crud.client_spec.ClientSpec().get_client_spec()
