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
import logging
from collections.abc import Iterator
from http import HTTPStatus

import fastapi
import pydantic.v1
import pytest
from fastapi.exception_handlers import http_exception_handler
from fastapi.testclient import TestClient

from mlrun.utils import logger
from mlrun.utils.logger import Logger, create_logger


class Handled1Error(Exception):
    pass


class Handled2Error(Exception):
    pass


class UnhandledError(Exception):
    pass


async def handler_returning_response(request: fastapi.Request, exc: Handled1Error):
    logger.warning("Handler caught Handled1Error exception, returning 204 response")
    return fastapi.Response(status_code=HTTPStatus.NO_CONTENT.value)


async def handler_returning_http_exception(
    request: fastapi.Request, exc: Handled2Error
):
    logger.warning(
        "Handler caught Handled2Error exception, returning HTTPException with 401"
    )
    return await http_exception_handler(
        request, fastapi.HTTPException(status_code=HTTPStatus.UNAUTHORIZED.value)
    )


test_router = fastapi.APIRouter()


@test_router.get("/success")
def success():
    logger.info("Success endpoint received request, returning 202")
    return fastapi.Response(status_code=202)


@test_router.get("/handled_1_error")
def handled_1_error():
    logger.info(
        "handled_exception_1 endpoint received request, raising handled 1 error"
    )
    raise Handled1Error("handled 1 error")


@test_router.get("/handled_2_error")
def handled_2_error():
    logger.info(
        "handled_exception_2 endpoint received request, raising handled 2 error"
    )
    raise Handled2Error("handled 2 error")


@test_router.get("/unhandled_exception")
def unhandled_exception():
    logger.info("unhandled endpoint received request, raising unhandled exception")
    raise UnhandledError("Unhandled exception")


class SomeScheme(pydantic.v1.BaseModel):
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
def client(
    request: pytest.FixtureRequest, app: fastapi.FastAPI
) -> Iterator[TestClient]:
    app.add_exception_handler(Handled1Error, handler_returning_response)
    app.add_exception_handler(Handled2Error, handler_returning_http_exception)

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
def stream_logger() -> Iterator[tuple[io.StringIO, Logger]]:
    stream = io.StringIO()
    stream_logger = create_logger("debug", name="test-logger", stream=stream)
    yield stream, stream_logger


@pytest.mark.parametrize(
    "log_config",
    [
        ({"mlrun_pipelines.imports": "DEBUG"}),
        ({"mlrun_pipelines.imports": "INFO"}),
        ({"mlrun_pipelines.imports": "WARNING"}),
        ({"mlrun_pipelines.imports": "ERROR"}),
        ({"mlrun_pipelines.imports": "CRITICAL"}),
    ],
)
def test_set_and_get_log_level(log_config: dict[str, str], client: TestClient):
    payload = {"domain_to_levels": log_config, "recursive": False}
    response = client.post("/api/_internal/log_levels", json=payload)
    assert response.status_code == 200

    # Get current log levels
    response = client.get("/api/_internal/log_levels")
    assert response.status_code == 200
    data = response.json()
    assert (
        data["domain_to_levels"].get("mlrun_pipelines.imports")
        == log_config["mlrun_pipelines.imports"]
    )
    # GET endpoint returns recursive as False by default
    assert data.get("recursive") is False


@pytest.mark.parametrize(
    "invalid_log_config",
    [
        ({"invalid_domain.imports": "INFO"}),  # Domain not starting with 'mlrun'
        ({"mlrun_pipelines.imports": "INVALID_LEVEL"}),  # Invalid log level
        ({"mlrun_pipelines.imports": 123}),  # Non-string log level
    ],
)
def test_invalid_log_config(invalid_log_config: dict[str, str], client: TestClient):
    payload = {"domain_to_levels": invalid_log_config, "recursive": False}
    response = client.post("/api/_internal/log_levels", json=payload)
    # Expecting a validation error (422 Unprocessable Entity)
    assert response.status_code == 422


def test_recursive_set_log_levels(client: TestClient):
    domain = "mlrun"
    sub_logger_name = "mlrun.api"
    logger = logging.getLogger(domain)
    sub_logger = logging.getLogger(sub_logger_name)

    # Ensure the loggers are in a known state.
    logger.setLevel(logging.NOTSET)
    sub_logger.setLevel(logging.NOTSET)

    # Set the domain log level recursively.
    payload = {"domain_to_levels": {domain: "ERROR"}, "recursive": True}
    response = client.post("/api/_internal/log_levels", json=payload)
    assert response.status_code == 200

    # Check that both the logger and its sub-logger have been updated.
    numeric_level = getattr(logging, "ERROR")
    assert logger.getEffectiveLevel() == numeric_level
    assert sub_logger.getEffectiveLevel() == numeric_level


def _ensure_request_logged(log_stream, verify_unhandled_exception: bool = False):
    lines = log_stream.getvalue().splitlines()
    assert "Received request" in lines[0]
    assert "Sending response" in lines[1]
    if verify_unhandled_exception:
        assert "Request handling failed" in lines[1]
