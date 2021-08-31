from fastapi import APIRouter

import mlrun.api.crud
import mlrun.api.schemas

router = APIRouter()


@router.get(
    "/healthz", response_model=mlrun.api.schemas.ClientSpec,
)
def health():

    # TODO: From 0.7.0 client uses the /client-spec endpoint,
    #  when this is the oldest relevant client, remove this logic from the healthz endpoint
    return mlrun.api.crud.ClientSpec().get_client_spec()
