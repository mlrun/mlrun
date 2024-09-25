# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import typing

import fastapi
import semver

import mlrun.common.schemas
import mlrun.runtimes
import mlrun.runtimes.utils
import mlrun.utils.helpers
import server.api.api.deps
import server.api.utils.builder
import server.api.utils.clients.iguazio
import server.api.utils.runtimes.nuclio
from mlrun.config import config
from mlrun.platforms import is_iguazio_session_cookie
from server.api.api.utils import get_allowed_path_prefixes_list

router = fastapi.APIRouter()


@router.get(
    "/frontend-spec",
    response_model=mlrun.common.schemas.FrontendSpec,
)
def get_frontend_spec(
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    jobs_dashboard_url = None
    model_monitoring_dashboard_url = None
    session = auth_info.session
    if session and is_iguazio_session_cookie(session):
        jobs_dashboard_url = _resolve_jobs_dashboard_url(session)
        model_monitoring_dashboard_url = _resolve_model_monitoring_dashboard_url(
            session
        )
    feature_flags = _resolve_feature_flags()
    registry, repository = mlrun.utils.helpers.get_parsed_docker_registry()
    repository = mlrun.utils.helpers.get_docker_repository_or_default(repository)
    function_deployment_target_image_template = (
        mlrun.runtimes.utils.fill_function_image_name_template(
            f"{registry}/",
            repository,
            "{project}",
            "{name}",
            "{tag}",
        )
    )
    registries_to_enforce_prefix = mlrun.runtimes.utils.resolve_function_target_image_registries_to_enforce_prefix()
    function_target_image_name_prefix_template = (
        config.httpdb.builder.function_target_image_name_prefix_template
    )
    return mlrun.common.schemas.FrontendSpec(
        jobs_dashboard_url=jobs_dashboard_url,
        model_monitoring_dashboard_url=model_monitoring_dashboard_url,
        abortable_function_kinds=mlrun.runtimes.RuntimeKinds.abortable_runtimes(),
        feature_flags=feature_flags,
        default_function_priority_class_name=config.default_function_priority_class_name,
        valid_function_priority_class_names=config.get_valid_function_priority_class_names(),
        default_function_image_by_kind=mlrun.mlconf.function_defaults.image_by_kind.to_dict(),
        function_deployment_target_image_template=function_deployment_target_image_template,
        function_deployment_target_image_name_prefix_template=function_target_image_name_prefix_template,
        function_deployment_target_image_registries_to_enforce_prefix=registries_to_enforce_prefix,
        function_deployment_mlrun_requirement=server.api.utils.builder.resolve_mlrun_install_command_version(),
        auto_mount_type=config.storage.auto_mount_type,
        auto_mount_params=config.get_storage_auto_mount_params(),
        default_artifact_path=config.artifact_path,
        default_function_pod_resources=mlrun.mlconf.default_function_pod_resources.to_dict(),
        default_function_preemption_mode=mlrun.mlconf.function_defaults.preemption_mode,
        feature_store_data_prefixes=config.feature_store.data_prefixes.to_dict(),
        allowed_artifact_path_prefixes_list=get_allowed_path_prefixes_list(),
        ce=config.ce.to_dict(),
        internal_labels=config.internal_labels(),
        artifact_limits=mlrun.common.schemas.ArtifactLimits(
            max_chunk_size=config.artifacts.limits.max_chunk_size,
            max_preview_size=config.artifacts.limits.max_preview_size,
            max_download_size=config.artifacts.limits.max_download_size,
        ),
    )


def try_get_grafana_service_url(session):
    if mlrun.mlconf.grafana_url:
        return mlrun.mlconf.grafana_url
    else:
        iguazio_client = server.api.utils.clients.iguazio.Client()
        return iguazio_client.try_get_grafana_service_url(session)


def _resolve_jobs_dashboard_url(session: str) -> typing.Optional[str]:
    grafana_service_url = try_get_grafana_service_url(session)
    if grafana_service_url:
        # FIXME: this creates a heavy coupling between mlrun and the grafana dashboard (name and filters) + org id
        return (
            grafana_service_url
            + "/d/mlrun-jobs-monitoring/mlrun-jobs-monitoring?orgId=1&var-groupBy={filter_name}"
            "&var-filter={filter_value}"
        )
    return None


def _resolve_model_monitoring_dashboard_url(session: str) -> typing.Optional[str]:
    grafana_service_url = try_get_grafana_service_url(session)
    if grafana_service_url:
        return grafana_service_url + (
            "/d/AohIXhAMk/model-monitoring-details?var-PROJECT={project}"
            "&var-MODELENDPOINT={model_endpoint}"
        )
    return None


def _resolve_feature_flags() -> mlrun.common.schemas.FeatureFlags:
    project_membership = mlrun.common.schemas.ProjectMembershipFeatureFlag.disabled
    if mlrun.mlconf.httpdb.authorization.mode == "opa":
        project_membership = mlrun.common.schemas.ProjectMembershipFeatureFlag.enabled
    authentication = mlrun.common.schemas.AuthenticationFeatureFlag(
        mlrun.mlconf.httpdb.authentication.mode
    )
    nuclio_streams = mlrun.common.schemas.NuclioStreamsFeatureFlag.disabled

    if mlrun.mlconf.get_parsed_igz_version() and semver.VersionInfo.parse(
        server.api.utils.runtimes.nuclio.resolve_nuclio_version()
    ) >= semver.VersionInfo.parse("1.7.8"):
        nuclio_streams = mlrun.common.schemas.NuclioStreamsFeatureFlag.enabled

    preemption_nodes = mlrun.common.schemas.PreemptionNodesFeatureFlag.disabled
    if mlrun.mlconf.is_preemption_nodes_configured():
        preemption_nodes = mlrun.common.schemas.PreemptionNodesFeatureFlag.enabled

    return mlrun.common.schemas.FeatureFlags(
        project_membership=project_membership,
        authentication=authentication,
        nuclio_streams=nuclio_streams,
        preemption_nodes=preemption_nodes,
    )
