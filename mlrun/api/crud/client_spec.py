import mlrun.api.schemas
import mlrun.utils.singleton
from mlrun.api.utils.clients import nuclio
from mlrun.config import config, default_config
from mlrun.runtimes.utils import resolve_mpijob_crd_version
from mlrun.utils import logger


class ClientSpec(metaclass=mlrun.utils.singleton.Singleton,):
    def __init__(self):
        self._cached_nuclio_version = None

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
            nuclio_version=self._resolve_nuclio_version(),
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
        )

    def _get_config_value_if_not_default(self, config_key):
        config_key_parts = config_key.split(".")
        current_config_value = config
        current_default_config_value = default_config
        for config_key_part in config_key_parts:
            current_config_value = getattr(current_config_value, config_key_part)
            current_default_config_value = current_default_config_value.get(
                config_key_part, ""
            )
        if current_config_value == current_default_config_value:
            return None
        else:
            return current_config_value

    # if nuclio version specified on mlrun config set it likewise,
    # if not specified, get it from nuclio api client
    # since this is a heavy operation (sending requests to API), and it's unlikely that the version
    # will change - cache it (this means if we upgrade nuclio, we need to restart mlrun to re-fetch the new version)
    def _resolve_nuclio_version(self):
        if not self._cached_nuclio_version:

            # config override everything
            nuclio_version = config.nuclio_version
            if not nuclio_version and config.nuclio_dashboard_url:
                try:
                    nuclio_client = nuclio.Client()
                    nuclio_version = nuclio_client.get_dashboard_version()
                except Exception as exc:
                    logger.warning("Failed to resolve nuclio version", exc=str(exc))

            self._cached_nuclio_version = nuclio_version

        return self._cached_nuclio_version
