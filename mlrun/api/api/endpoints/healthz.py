from fastapi import APIRouter

from mlrun.config import config, default_config
from mlrun.runtimes.utils import resolve_mpijob_crd_version

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
        # These have a default value, therefore we want to send them only if their value is not the default one
        # (otherwise clients don't know when to use server value and when to use client value)
        "ui_projects_prefix": _get_config_value_if_not_default("ui.projects_prefix"),
        "scrape_metrics": _get_config_value_if_not_default("scrape_metrics"),
        "hub_url": _get_config_value_if_not_default("hub_url"),
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
