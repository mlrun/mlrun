# Copyright 2025 Iguazio
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
import http
import logging
import os

import fastapi

import mlrun.common.schemas.internal.logging
import mlrun.errors

import services.discovery.service

router = fastapi.APIRouter(prefix="/log_levels")


def _apply_log_level_locally(
    cfg: mlrun.common.schemas.internal.logging.LogLevelMapping,
) -> None:
    for domain, level in cfg.domain_to_levels.items():
        num = getattr(logging, level.upper(), None)
        if num is None:
            raise mlrun.errors.MLRunBadRequestError(f"Invalid log level '{level}'")
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
    cfg: mlrun.common.schemas.internal.logging.LogLevelMapping,
    request: fastapi.Request,
):
    discovery_service = services.discovery.service.K8sServiceDiscovery(
        namespace=os.getenv("MLRUN_NAMESPACE"),
    )

    _apply_log_level_locally(cfg)
    await discovery_service.broadcast(
        excluded_services=["mlrun-api-chief"],
        path="/_internal/log_levels",
        json_payload=cfg.dict(),
        timeout=10.0,
        headers=dict(request.headers),
    )
    return fastapi.Response(status_code=http.HTTPStatus.OK.value)


@router.get(
    "",
    response_model=mlrun.common.schemas.internal.logging.LogLevelMapping,
)
async def get_log_levels():
    return mlrun.common.schemas.internal.logging.LogLevelMapping(
        domain_to_levels={
            n: logging.getLevelName(lg.getEffectiveLevel())
            for n, lg in logging.root.manager.loggerDict.items()
            if isinstance(lg, logging.Logger)
        },
        recursive=False,
    )
