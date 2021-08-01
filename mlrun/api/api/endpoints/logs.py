import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.opa

router = fastapi.APIRouter()


# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@router.post("/log/{project}/{uid}")
async def store_log(
    request: fastapi.Request,
    project: str,
    uid: str,
    append: bool = True,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_verifier.auth_info,
    )
    body = await request.body()
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Logs().store_log,
        body,
        project,
        uid,
        append,
    )
    return {}


# curl http://localhost:8080/log/prj/7
@router.get("/log/{project}/{uid}")
def get_log(
    project: str,
    uid: str,
    size: int = -1,
    offset: int = 0,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = fastapi.Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    run_state, log = mlrun.api.crud.Logs().get_logs(
        db_session, project, uid, size, offset
    )
    headers = {
        "x-mlrun-run-state": run_state,
        # pod_status was changed x-mlrun-run-state in 0.5.3, keeping it here for backwards compatibility (so <0.5.3
        # clients will work with the API)
        # TODO: remove this in 0.7.0
        "pod_status": run_state,
    }
    return fastapi.Response(content=log, media_type="text/plain", headers=headers)
