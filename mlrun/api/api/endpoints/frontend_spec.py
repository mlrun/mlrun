import typing

import fastapi

import mlrun.api.api.deps
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.builder
import mlrun.runtimes
import mlrun.runtimes.utils
import mlrun.utils.helpers
from mlrun.config import config

router = fastapi.APIRouter()


@router.get(
    "/frontend-spec", response_model=mlrun.api.schemas.FrontendSpec,
)
def get_frontend_spec(
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    # In Iguazio 3.0 auth is turned off, but for this endpoint specifically the session is a must, so getting it from
    # the cookie like it was before
    # TODO: remove when Iguazio 3.0 is no longer relevant
    session: typing.Optional[str] = fastapi.Cookie(None),
):
    jobs_dashboard_url = None
    session = auth_info.session or session
    if session:
        jobs_dashboard_url = _resolve_jobs_dashboard_url(session)
    feature_flags = _resolve_feature_flags()
    registry, repository = mlrun.utils.helpers.get_parsed_docker_registry()
    repository = mlrun.utils.helpers.get_docker_repository_or_default(repository)
    function_deployment_target_image_template = mlrun.runtimes.utils.fill_function_image_name_template(
        f"{registry}/", repository, "{project}", "{name}", "{tag}",
    )
    return mlrun.api.schemas.FrontendSpec(
        jobs_dashboard_url=jobs_dashboard_url,
        abortable_function_kinds=mlrun.runtimes.RuntimeKinds.abortable_runtimes(),
        feature_flags=feature_flags,
        default_function_priority_class_name=config.default_function_priority_class_name,
        valid_function_priority_class_names=config.get_valid_function_priority_class_names(),
        default_function_image_by_kind=mlrun.mlconf.function_defaults.image_by_kind.to_dict(),
        function_deployment_target_image_template=function_deployment_target_image_template,
        function_deployment_mlrun_command=mlrun.builder.resolve_mlrun_install_command(),
        auto_mount_type=config.storage.auto_mount_type,
        auto_mount_params=config.get_storage_auto_mount_params(),
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


def _resolve_feature_flags() -> mlrun.api.schemas.FeatureFlags:
    project_membership = mlrun.api.schemas.ProjectMembershipFeatureFlag.disabled
    if mlrun.mlconf.httpdb.authorization.mode == "opa":
        project_membership = mlrun.api.schemas.ProjectMembershipFeatureFlag.enabled
    authentication = mlrun.api.schemas.AuthenticationFeatureFlag(
        mlrun.mlconf.httpdb.authentication.mode
    )
    return mlrun.api.schemas.FeatureFlags(
        project_membership=project_membership, authentication=authentication
    )
