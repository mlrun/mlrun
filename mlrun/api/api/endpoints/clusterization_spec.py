import fastapi

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.chief
from mlrun.utils import logger

router = fastapi.APIRouter()


@router.get("/clusterization-spec", response_model=mlrun.api.schemas.ClusterizationSpec)
def clusterization_spec():
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting clusterization spec from worker, re-routing to chief",
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return chief_client.get_clusterization_spec()

    return mlrun.api.crud.ClusterizationSpec().get_clusterization_spec()
