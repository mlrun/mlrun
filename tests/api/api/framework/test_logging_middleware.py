# Copyright 2023 Iguazio
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
import io
import typing
from http import HTTPStatus

import fastapi
import pydantic
import pytest
from fastapi.exception_handlers import http_exception_handler
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun.utils import logger
from mlrun.utils.logger import Logger, create_logger
from server.api.main import app


class Handled1Error(Exception):
    pass


class Handled2Error(Exception):
    pass


class UnhandledError(Exception):
    pass


@app.exception_handler(Handled1Error)
async def handler_returning_response(request: fastapi.Request, exc: Handled1Error):
    logger.warning("Handler caught HandledException1 exception, returning 204 response")
    return fastapi.Response(status_code=HTTPStatus.NO_CONTENT.value)


@app.exception_handler(Handled1Error)
async def handler_returning_http_exception(
    request: fastapi.Request, exc: Handled1Error
):
    logger.warning(
        "Handler caught HandledException2 exception, returning HTTPException with 401"
    )
    return await http_exception_handler(
        request, fastapi.HTTPException(status_code=HTTPStatus.UNAUTHORIZED.value)
    )


test_router = fastapi.APIRouter()


@test_router.get("/success")
def success():
    logger.info("Success endpoint received request, returning 202")
    return fastapi.Response(status_code=202)


@test_router.get("/handled_exception_1")
def handled_exception_1():
    logger.info(
        "handled_exception_1 endpoint received request, raising handled exception 1"
    )
    raise Handled1Error("handled exception 1")


@test_router.get("/handled_exception_2")
def handled_exception_2():
    logger.info(
        "handled_exception_2 endpoint received request, raising handled exception 2"
    )
    raise Handled1Error("handled exception 2")


@test_router.get("/unhandled_exception")
def unhandled_exception():
    logger.info("unhandled endpoint received request, raising unhandled exception")
    raise UnhandledError("Unhandled exception")


class SomeScheme(pydantic.BaseModel):
    id: str


@test_router.post("/fastapi_handled_exception")
def fastapi_handled_exception(model: SomeScheme):
    logger.info("Should not get here, will fail on body validation")


middleware_modes = [
    "with_middleware",
    "without_middleware",
]


# must add it here since we're adding routes
@pytest.fixture(params=middleware_modes)
def client(request) -> typing.Generator:
    # save a copy of the middlewares. we would want to restore them once we're done with the test
    user_middleware = app.user_middleware.copy()
    try:
        if request.param == "without_middleware":
            # this overrides the webapp middlewares by removing the logging middleware
            app.user_middleware = []
            app.middleware_stack = app.build_middleware_stack()
        app.include_router(test_router, prefix="/test")
        with TestClient(app) as c:
            yield c
    finally:
        # restore back the middlewares
        if request.param == "without_middleware":
            app.user_middleware = user_middleware
            app.middleware_stack = app.build_middleware_stack()


@pytest.fixture
def stream_logger(request) -> (io.StringIO, Logger):
    stream = io.StringIO()
    stream_logger = create_logger("debug", name="test-logger", stream=stream)
    yield stream, stream_logger


def test_logging_middleware(db: Session, client: TestClient, stream_logger) -> None:
    stream, logger_instance = stream_logger
    stream: io.StringIO
    has_logger_middleware = False
    for middleware in client.app.user_middleware:
        if "logger" in middleware.kwargs:
            middleware.kwargs["logger"] = logger_instance
            has_logger_middleware = True
    client.app.middleware_stack = client.app.build_middleware_stack()

    resp = client.get("/test/success")
    assert resp.status_code == HTTPStatus.ACCEPTED.value
    if has_logger_middleware:
        _ensure_request_logged(stream)
        stream.seek(0)

    resp = client.get("/test/handled_exception_1")
    assert resp.status_code == HTTPStatus.NO_CONTENT.value
    if has_logger_middleware:
        _ensure_request_logged(stream)
        stream.seek(0)

    resp = client.get("/test/handled_exception_2")
    assert resp.status_code == HTTPStatus.UNAUTHORIZED.value
    if has_logger_middleware:
        _ensure_request_logged(stream)
        stream.seek(0)

    resp = client.post("/test/fastapi_handled_exception")
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY.value
    if has_logger_middleware:
        _ensure_request_logged(stream)
        stream.seek(0)

    with pytest.raises(UnhandledError):
        # In a real fastapi (and not test) unhandled exception returns 500
        client.get("/test/unhandled_exception")
    if has_logger_middleware:
        _ensure_request_logged(stream, verify_unhandled_exception=True)
        stream.seek(0)


def _ensure_request_logged(log_stream, verify_unhandled_exception: bool = False):
    lines = log_stream.getvalue().splitlines()
    assert "Received request" in lines[0]
    assert "Sending response" in lines[1]
    if verify_unhandled_exception:
        assert "Request handling failed" in lines[1]
