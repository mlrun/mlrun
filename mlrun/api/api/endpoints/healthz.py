from os import environ

from fastapi import APIRouter

from mlrun.config import config

router = APIRouter()


@router.get("/healthz")
def health():
    return {
        "version": config.version,
        "namespace": config.namespace,
        "docker_registry": environ.get('DEFAULT_DOCKER_REGISTRY', ''),
        "remote_host": config.remote_host,
        "ui_url": config.ui_url,
        "artifact_path": config.artifact_path,
    }
