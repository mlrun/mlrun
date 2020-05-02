from fastapi import APIRouter

from mlrun.config import config

router = APIRouter()


@router.get("/healthz")
def health():
    return {
        "version": config.version,
    }
