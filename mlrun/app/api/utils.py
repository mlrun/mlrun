from http import HTTPStatus
from pathlib import Path

from fastapi import HTTPException

from mlrun.app.main import logs_dir
from mlrun.utils import logger


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    logger.error(str(kw))
    raise HTTPException(status_code=status, detail=kw)


def log_path(project, uid) -> Path:
    return logs_dir / project / uid
