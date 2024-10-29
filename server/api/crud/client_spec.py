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
import mlrun.common.schemas
import mlrun.utils.singleton
import server.api.runtime_handlers.mpijob
import server.api.utils.runtimes.nuclio
from mlrun.config import Config, config, default_config


class ClientSpec(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def get_client_spec(
        self, client_version: str = None, client_python_version: str = None
    ) -> mlrun.common.schemas.ClientSpec:
        mpijob_crd_version = (
            server.api.runtime_handlers.mpijob.resolve_mpijob_crd_version()
        )

        return mlrun.common.schemas.ClientSpec(
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
            kfp_image=self._resolve_image_by_client_versions(
                config.kfp_image, client_version, client_python_version
            ),
            kfp_url=config.kfp_url,
            dask_kfp_image=self._resolve_image_by_client_versions(
                config.dask_kfp_image, client_version, client_python_version
            ),
            api_url=config.httpdb.api_url,
            nuclio_version=server.api.utils.runtimes.nuclio.resolve_nuclio_version(),
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
            ce=config.ce.to_dict(),
            logs=self._get_config_value_if_not_default("httpdb.logs"),
            feature_store_data_prefixes=self._get_config_value_if_not_default(
                "feature_store.data_prefixes"
            ),
            feature_store_default_targets=self._get_config_value_if_not_default(
                "feature_store.default_targets"
            ),
            external_platform_tracking=self._get_config_value_if_not_default(
                "external_platform_tracking"
            ),
            model_endpoint_monitoring_endpoint_store_connection=self._get_config_value_if_not_default(
                "model_endpoint_monitoring.endpoint_store_connection"
            ),
            model_monitoring_tsdb_connection=self._get_config_value_if_not_default(
                "model_endpoint_monitoring.tsdb_connection"
            ),
            model_monitoring_stream_connection=self._get_config_value_if_not_default(
                "model_endpoint_monitoring.stream_connection"
            ),
            packagers=self._get_config_value_if_not_default("packagers"),
            alerts_mode=self._get_config_value_if_not_default("alerts.mode"),
        )

    @staticmethod
    def _resolve_image_by_client_versions(
        image: str, client_version: str = None, client_python_version=None
    ):
        """
        This method main purpose is to provide enriched images for deployment processes which are being executed on
        client side, such as building a workflow. The whole enrichment and construction of a workflow is being done on
        client side unlike submitting job where the main enrichment and construction of the resource runtime is being
        applied on the backend side. Therefore for the workflow case we need to provide it with already enriched
        images.
        :param image: image name
        :param client_version: the client mlrun version
        :param client_python_version: the client python version
        :return: enriched image url
        """
        try:
            return mlrun.utils.helpers.enrich_image_url(
                image, client_version, client_python_version
            )
        # if for some reason the user provided un-parsable versions, fall back to resolve version only by server
        except ValueError:
            return mlrun.utils.helpers.enrich_image_url(image)

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
