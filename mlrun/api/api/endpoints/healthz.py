from fastapi import APIRouter

from mlrun.api.utils.clients import nuclio
from mlrun.config import config, default_config
from mlrun.runtimes.utils import resolve_mpijob_crd_version
from mlrun.utils import logger

router = APIRouter()


@router.get("/healthz")
def health():
    mpijob_crd_version = resolve_mpijob_crd_version(api_context=True)
    return {
        "version": config.version,
        "namespace": config.namespace,
        "docker_registry": config.httpdb.builder.docker_registry,
        "remote_host": config.remote_host,
        "mpijob_crd_version": mpijob_crd_version,
        "ui_url": config.resolve_ui_url(),
        "artifact_path": config.artifact_path,
        "spark_app_image": config.spark_app_image,
        "spark_app_image_tag": config.spark_app_image_tag,
        "kfp_image": config.kfp_image,
        "dask_kfp_image": config.dask_kfp_image,
        "api_url": config.httpdb.api_url,
        "nuclio_version": _resolve_nuclio_version(),
        # These have a default value, therefore we want to send them only if their value is not the default one
        # (otherwise clients don't know when to use server value and when to use client value)
        "ui_projects_prefix": _get_config_value_if_not_default("ui.projects_prefix"),
        "scrape_metrics": _get_config_value_if_not_default("scrape_metrics"),
        "hub_url": _get_config_value_if_not_default("hub_url"),
        "default_function_node_selector": _get_config_value_if_not_default(
            "default_function_node_selector"
        ),
    }


def _get_config_value_if_not_default(config_key):
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


cached_nuclio_version = None


# if nuclio version specified on mlrun config set it likewise,
# if not specified, get it from nuclio api client
# since this is a heavy operation (sending requests to API), and it's unlikely that the version
# will change - cache it (this means if we upgrade nuclio, we need to restart mlrun to re-fetch the new version)
def _resolve_nuclio_version():
    global cached_nuclio_version
    if not cached_nuclio_version:

        # config override everything
        nuclio_version = config.nuclio_version
        if not nuclio_version and config.nuclio_dashboard_url:
            try:
                nuclio_client = nuclio.Client()
                nuclio_version = nuclio_client.get_dashboard_version()
            except Exception as exc:
                logger.warning("Failed to resolve nuclio version", exc=str(exc))

        cached_nuclio_version = nuclio_version

    return cached_nuclio_version
