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


import pydantic.v1

from .function import Function
from .k8s import Resources


class ClientSpec(pydantic.v1.BaseModel):
    version: str | None
    namespace: str | None
    docker_registry: str | None
    remote_host: str | None
    mpijob_crd_version: str | None
    ui_url: str | None
    artifact_path: str | None
    feature_store_data_prefixes: dict[str, str] | None
    feature_store_default_targets: str | None
    spark_app_image: str | None
    spark_app_image_tag: str | None
    spark_history_server_path: str | None
    spark_operator_version: str | None
    kfp_image: str | None
    kfp_url: str | None
    kfp_default_workflow_timeout: str | None
    dask_kfp_image: str | None
    api_url: str | None
    nuclio_version: str | None
    ui_projects_prefix: str | None
    scrape_metrics: str | None
    default_function_node_selector: str | None
    igz_version: str | None
    auto_mount_type: str | None
    auto_mount_params: str | None
    default_function_priority_class_name: str | None
    valid_function_priority_class_names: str | None
    default_tensorboard_logs_path: str | None
    default_function_pod_resources: Resources | None
    preemptible_nodes_node_selector: str | None
    preemptible_nodes_tolerations: str | None
    default_preemption_mode: str | None
    force_run_local: str | None
    function: Function | None
    redis_url: str | None
    redis_type: str | None
    sql_url: str | None
    ce: dict | None
    # not passing them as one object as it possible client user would like to override only one of the params
    calculate_artifact_hash: str | None
    generate_artifact_target_path_from_artifact_hash: str | None
    logs: dict | None
    packagers: dict | None
    external_platform_tracking: dict | None
    alerts_mode: str | None
    system_id: str | None
    model_endpoint_monitoring_store_prefixes: dict[str, str] | None
    authentication_mode: str | None
    # Iguazio V4 OAuth token provider configuration
    oauth_internal_token_endpoint: str | None
    oauth_external_token_endpoint: str | None
    authorization_namespaces_resources: str | None
    default_runtime_image_by_kind: dict[str, str] | None
