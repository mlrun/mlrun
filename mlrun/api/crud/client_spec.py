# Copyright 2018 Iguazio
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
import mlrun.api.schemas
import mlrun.utils.singleton
from mlrun.config import Config, config, default_config
from mlrun.runtimes.utils import resolve_mpijob_crd_version, resolve_nuclio_version


class ClientSpec(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def get_client_spec(self):
        mpijob_crd_version = resolve_mpijob_crd_version(api_context=True)
        return mlrun.api.schemas.ClientSpec(
            version=config.version,
            namespace=config.namespace,
            docker_registry=config.httpdb.builder.docker_registry,
            remote_host=config.remote_host,
            mpijob_crd_version=mpijob_crd_version,
            ui_url=config.resolve_ui_url(),
            artifact_path=config.artifact_path,
            spark_app_image=config.spark_app_image,
            spark_app_image_tag=config.spark_app_image_tag,
            spark_history_server_path=config.spark_history_server_path,
            kfp_image=config.kfp_image,
            kfp_url=config.resolve_kfp_url(),
            dask_kfp_image=config.dask_kfp_image,
            api_url=config.httpdb.api_url,
            nuclio_version=resolve_nuclio_version(),
            spark_operator_version=config.spark_operator_version,
            calculate_artifact_hash=config.artifacts.calculate_hash,
            generate_artifact_target_path_from_artifact_hash=config.artifacts.generate_target_path_from_artifact_hash,
            redis_url=config.redis.url,
            redis_type=config.redis.type,
            sql_url=config.sql.url,
            # These don't have a default value, but we don't send them if they are not set to allow the client to know
            # when to use server value and when to use client value (server only if set). Since their default value is
            # empty and not set is also empty we can use the same _get_config_value_if_not_default
            default_function_priority_class_name=self._get_config_value_if_not_default(
                "default_function_priority_class_name"
            ),
            valid_function_priority_class_names=self._get_config_value_if_not_default(
                "valid_function_priority_class_names"
            ),
            # These have a default value, therefore we want to send them only if their value is not the default one
            # (otherwise clients don't know when to use server value and when to use client value)
            ui_projects_prefix=self._get_config_value_if_not_default(
                "ui.projects_prefix"
            ),
            scrape_metrics=self._get_config_value_if_not_default("scrape_metrics"),
            hub_url=self._get_config_value_if_not_default("hub_url"),
            default_function_node_selector=self._get_config_value_if_not_default(
                "default_function_node_selector"
            ),
            igz_version=self._get_config_value_if_not_default("igz_version"),
            auto_mount_type=self._get_config_value_if_not_default(
                "storage.auto_mount_type"
            ),
            auto_mount_params=self._get_config_value_if_not_default(
                "storage.auto_mount_params"
            ),
            default_tensorboard_logs_path=self._get_config_value_if_not_default(
                "default_tensorboard_logs_path"
            ),
            default_function_pod_resources=self._get_config_value_if_not_default(
                "default_function_pod_resources"
            ),
            preemptible_nodes_node_selector=self._get_config_value_if_not_default(
                "preemptible_nodes.node_selector"
            ),
            preemptible_nodes_tolerations=self._get_config_value_if_not_default(
                "preemptible_nodes.tolerations"
            ),
            default_preemption_mode=self._get_config_value_if_not_default(
                "function_defaults.preemption_mode"
            ),
            force_run_local=self._get_config_value_if_not_default("force_run_local"),
            function=self._get_config_value_if_not_default("function"),
            # ce_mode is deprecated, we will use the full ce config instead and ce_mode will be removed in 1.6.0
            ce_mode=config.ce.mode,
            ce=config.ce.to_dict(),
            logs=self._get_config_value_if_not_default("httpdb.logs"),
            feature_store_data_prefixes=self._get_config_value_if_not_default(
                "feature_store.data_prefixes"
            ),
        )

    @staticmethod
    def _get_config_value_if_not_default(config_key):
        config_key_parts = config_key.split(".")
        current_config_value = config
        current_default_config_value = default_config
        for config_key_part in config_key_parts:
            current_config_value = getattr(current_config_value, config_key_part)
            current_default_config_value = current_default_config_value.get(
                config_key_part, ""
            )
        # when accessing attribute in Config, if the object is of type Mapping it returns the object in type Config
        if isinstance(current_config_value, Config):
            current_config_value = current_config_value.to_dict()
        if current_config_value == current_default_config_value:
            return None
        else:
            return current_config_value
