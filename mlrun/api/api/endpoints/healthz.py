from os import environ

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
        "docker_registry": environ.get('DEFAULT_DOCKER_REGISTRY', ''),
        "remote_host": config.remote_host,
        "mpijob_crd_version": mpijob_crd_version,
        "ui_url": config.ui_url,
        "artifact_path": config.artifact_path,
    }
