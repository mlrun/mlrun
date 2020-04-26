from http import HTTPStatus

from fastapi import HTTPException

from mlrun.utils import logger


def json_error(status=HTTPStatus.BAD_REQUEST, **kw):
    logger.error(str(kw))
    raise HTTPException(status_code=status, detail=kw)
