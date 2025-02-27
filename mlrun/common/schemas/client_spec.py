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

import pydantic.v1

from .function import Function
from .k8s import Resources


class ClientSpec(pydantic.v1.BaseModel):
    version: typing.Optional[str]
    namespace: typing.Optional[str]
    docker_registry: typing.Optional[str]
    remote_host: typing.Optional[str]
    mpijob_crd_version: typing.Optional[str]
    ui_url: typing.Optional[str]
    artifact_path: typing.Optional[str]
    feature_store_data_prefixes: typing.Optional[dict[str, str]]
    feature_store_default_targets: typing.Optional[str]
    spark_app_image: typing.Optional[str]
    spark_app_image_tag: typing.Optional[str]
    spark_history_server_path: typing.Optional[str]
    spark_operator_version: typing.Optional[str]
    kfp_image: typing.Optional[str]
    kfp_url: typing.Optional[str]
    dask_kfp_image: typing.Optional[str]
    api_url: typing.Optional[str]
    nuclio_version: typing.Optional[str]
    ui_projects_prefix: typing.Optional[str]
    scrape_metrics: typing.Optional[str]
    default_function_node_selector: typing.Optional[str]
    igz_version: typing.Optional[str]
    auto_mount_type: typing.Optional[str]
    auto_mount_params: typing.Optional[str]
    default_function_priority_class_name: typing.Optional[str]
    valid_function_priority_class_names: typing.Optional[str]
    default_tensorboard_logs_path: typing.Optional[str]
    default_function_pod_resources: typing.Optional[Resources]
    preemptible_nodes_node_selector: typing.Optional[str]
    preemptible_nodes_tolerations: typing.Optional[str]
    default_preemption_mode: typing.Optional[str]
    force_run_local: typing.Optional[str]
    function: typing.Optional[Function]
    redis_url: typing.Optional[str]
    redis_type: typing.Optional[str]
    sql_url: typing.Optional[str]
    ce: typing.Optional[dict]
    # not passing them as one object as it possible client user would like to override only one of the params
    calculate_artifact_hash: typing.Optional[str]
    generate_artifact_target_path_from_artifact_hash: typing.Optional[str]
    logs: typing.Optional[dict]
    packagers: typing.Optional[dict]
    external_platform_tracking: typing.Optional[dict]
    alerts_mode: typing.Optional[str]
    system_id: typing.Optional[str]
    model_endpoint_monitoring_store_prefixes: typing.Optional[dict[str, str]]
