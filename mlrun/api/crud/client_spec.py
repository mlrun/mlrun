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
            dask_kfp_image=config.dask_kfp_image,
            api_url=config.httpdb.api_url,
            nuclio_version=resolve_nuclio_version(),
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
            spark_operator_version=self._get_config_value_if_not_default(
                "spark_operator_version"
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
            force_run_local=config.force_run_local,
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
