import typing

import fastapi

import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio

router = fastapi.APIRouter()


@router.get(
    "/frontend-specs", response_model=mlrun.api.schemas.FrontendSpec,
)
def get_frontend_specs(session: typing.Optional[str] = fastapi.Cookie(None)):
    jobs_dashboard_url = None
    if session:
        jobs_dashboard_url = _resolve_jobs_dashboard_url(session)
    return mlrun.api.schemas.FrontendSpec(jobs_dashboard_url=jobs_dashboard_url)


def _resolve_jobs_dashboard_url(session: str):
    zebo_client = mlrun.api.utils.clients.iguazio.Client()
    grafana_service_url = zebo_client.get_grafana_service_url_if_exists(session)
    # FIXME: this creates a heavy coupling between mlrun and the dashboard + org id
    return (
        f"{grafana_service_url}/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1&var-groupBy={{filter_name}}"
        f"&var-filter={{filter_value}}"
    )
