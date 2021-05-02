import typing

import fastapi

import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.runtimes

router = fastapi.APIRouter()


@router.get(
    "/frontend-spec", response_model=mlrun.api.schemas.FrontendSpec,
)
def get_frontend_spec(session: typing.Optional[str] = fastapi.Cookie(None)):
    jobs_dashboard_url = None
    if session:
        jobs_dashboard_url = _resolve_jobs_dashboard_url(session)
    return mlrun.api.schemas.FrontendSpec(
        jobs_dashboard_url=jobs_dashboard_url,
        abortable_function_kinds=mlrun.runtimes.RuntimeKinds.abortable_runtimes(),
    )


def _resolve_jobs_dashboard_url(session: str) -> typing.Optional[str]:
    iguazio_client = mlrun.api.utils.clients.iguazio.Client()
    grafana_service_url = iguazio_client.try_get_grafana_service_url(session)
    if grafana_service_url:
        # FIXME: this creates a heavy coupling between mlrun and the grafana dashboard (name and filters) + org id
        return (
            f"{grafana_service_url}/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1&var-groupBy={{filter_name}}"
            f"&var-filter={{filter_value}}"
        )
    return None
