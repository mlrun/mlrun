# Copyright 2026 Iguazio
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


import typing

import pydantic

from .function import Function
from .k8s import Resources


def _accepts_str(annotation) -> bool:
    return annotation is str or str in typing.get_args(annotation)


class ClientSpec(pydantic.BaseModel):
    # ML-12736: pydantic v1 implicitly coerced bool/int/float into str-typed fields (e.g. the
    # bool config ``calculate_artifact_hash`` was serialized as "True"); pydantic v2 is strict.
    # Reproduce the v1 coercion so existing pydantic-v1 clients keep parsing these as strings.
    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_v1_str_fields(cls, data):
        if isinstance(data, dict):
            data = dict(data)  # don't mutate the caller's input
            for name, field in cls.model_fields.items():
                value = data.get(name)
                if (
                    value is not None
                    and isinstance(value, (bool, int, float))
                    and not isinstance(value, str)
                    and _accepts_str(field.annotation)
                ):
                    data[name] = str(value)
        return data

    version: str | None = None
    namespace: str | None = None
    docker_registry: str | None = None
    remote_host: str | None = None
    mpijob_crd_version: str | None = None
    ui_url: str | None = None
    artifact_path: str | None = None
    feature_store_data_prefixes: dict[str, str] | None = None
    feature_store_default_targets: str | None = None
    spark_app_image: str | None = None
    spark_app_image_tag: str | None = None
    spark_history_server_path: str | None = None
    spark_operator_version: str | None = None
    kfp_image: str | None = None
    kfp_url: str | None = None
    kfp_default_workflow_timeout: str | None = None
    dask_kfp_image: str | None = None
    api_url: str | None = None
    nuclio_version: str | None = None
    ui_projects_prefix: str | None = None
    scrape_metrics: str | None = None
    default_function_node_selector: str | None = None
    igz_version: str | None = None
    auto_mount_type: str | None = None
    auto_mount_params: str | None = None
    default_function_priority_class_name: str | None = None
    valid_function_priority_class_names: str | None = None
    default_tensorboard_logs_path: str | None = None
    default_function_pod_resources: Resources | None = None
    preemptible_nodes_node_selector: str | None = None
    preemptible_nodes_tolerations: str | None = None
    default_preemption_mode: str | None = None
    force_run_local: str | None = None
    function: Function | None = None
    redis_url: str | None = None
    redis_type: str | None = None
    sql_url: str | None = None
    ce: dict | None = None
    # not passing them as one object as it possible client user would like to override only one of the params
    calculate_artifact_hash: str | None = None
    generate_artifact_target_path_from_artifact_hash: str | None = None
    logs: dict | None = None
    packagers: dict | None = None
    external_platform_tracking: dict | None = None
    alerts_mode: str | None = None
    system_id: str | None = None
    model_endpoint_monitoring_store_prefixes: dict[str, str] | None = None
    authentication_mode: str | None = None
    # Iguazio V4 OAuth token provider configuration
    oauth_internal_token_endpoint: str | None = None
    oauth_external_token_endpoint: str | None = None
    authorization_namespaces_mlrun: str | None = None
    default_runtime_image_by_kind: dict[str, str] | None = None
    telemetry_enabled: bool | None = None
    telemetry_otlp_endpoint: str | None = None
    telemetry_insecure: bool | None = None
    telemetry_model_monitoring_interval: int | None = None
