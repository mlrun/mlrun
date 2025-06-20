# ---- logging endpoint -------------------------------------------------
import logging
import os

import fastapi
from fastapi import Request
from fastapi.openapi.models import Response

from mlrun.common.schemas.internal.logging import LogLevelMapping
from mlrun.errors import MLRunBadRequestError

import services.discovery.service

router = fastapi.APIRouter(prefix="/log_levels")
_discovery_service = services.discovery.service.K8sServiceDiscovery(
    namespace=os.getenv("MLRUN_NAMESPACE"),
)


def _apply_log_level_locally(
    cfg: LogLevelMapping,
) -> None:
    for domain, level in cfg.domain_to_levels.items():
        num = getattr(logging, level.upper(), None)
        if num is None:
            raise MLRunBadRequestError(f"Invalid log level '{level}'")
        if cfg.recursive:
            for name, logger in logging.root.manager.loggerDict.items():
                if isinstance(logger, logging.Logger) and (
                    name == domain or name.startswith(f"{domain}.")
                ):
                    logger.setLevel(num)
        else:
            logging.getLogger(domain).setLevel(num)


@router.post("")
async def set_log_levels(
    cfg: LogLevelMapping,
    request: Request,
):
    _apply_log_level_locally(cfg)
    await _discovery_service.broadcast(
        excluded_services=["mlrun-api-chief"],
        path="/_internal/log_levels",
        json_payload=cfg.dict(),
        timeout=10.0,
        headers=dict(request.headers),
    )
    return Response(status_code=200)


@router.get(
    "",
    response_model=LogLevelMapping,
)
async def get_log_levels():
    return LogLevelMapping(
        domain_to_levels={
            n: logging.getLevelName(lg.getEffectiveLevel())
            for n, lg in logging.root.manager.loggerDict.items()
            if isinstance(lg, logging.Logger)
        },
        recursive=False,
    )
