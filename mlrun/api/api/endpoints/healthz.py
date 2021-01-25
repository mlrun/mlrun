from fastapi import APIRouter

from mlrun.config import config
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
        "ui_projects_prefix": config.ui.projects_prefix,
        "artifact_path": config.artifact_path,
        "spark_app_image": config.spark_app_image,
        "spark_app_image_tag": config.spark_app_image_tag,
    }
