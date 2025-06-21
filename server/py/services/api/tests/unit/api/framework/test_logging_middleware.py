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

import collections.abc
import http
import logging

import fastapi
import fastapi.exception_handlers
import fastapi.testclient
import pydantic.v1
import pytest

import mlrun.utils
import mlrun.utils.logger

import services.discovery.service


class Handled1Error(Exception):
    pass


class Handled2Error(Exception):
    pass


class UnhandledError(Exception):
    pass


async def handler_returning_response(
    request: fastapi.Request,
    exc: Handled1Error,
) -> fastapi.Response:
    mlrun.utils.logger.warning(
        "Handler caught Handled1Error exception, returning 204 response"
    )
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)


async def handler_returning_http_exception(
    request: fastapi.Request,
    exc: Handled2Error,
) -> fastapi.Response:
    mlrun.utils.logger.warning(
        "Handler caught Handled2Error exception, returning HTTPException with 401"
    )
    return await fastapi.exception_handlers.http_exception_handler(
        request, fastapi.HTTPException(status_code=http.HTTPStatus.UNAUTHORIZED.value)
    )


test_router = fastapi.APIRouter()


@test_router.get("/success")
def success():
    mlrun.utils.logger.info("Success endpoint received request, returning 202")
    return fastapi.Response(status_code=202)


@test_router.get("/handled_1_error")
def handled_1_error():
    mlrun.utils.logger.info(
        "handled_exception_1 endpoint received request, raising handled 1 error"
    )
    raise Handled1Error("handled 1 error")


@test_router.get("/handled_2_error")
def handled_2_error():
    mlrun.utils.logger.info(
        "handled_exception_2 endpoint received request, raising handled 2 error"
    )
    raise Handled2Error("handled 2 error")


@test_router.get("/unhandled_exception")
def unhandled_exception():
    mlrun.utils.logger.info(
        "unhandled endpoint received request, raising unhandled exception"
    )
    raise UnhandledError("Unhandled exception")


class SomeScheme(pydantic.v1.BaseModel):
    id: str


@test_router.post("/fastapi_handled_exception")
def fastapi_handled_exception(
    model: SomeScheme,
):
    mlrun.utils.logger.info("Should not get here, will fail on body validation")


middleware_modes = [
    "with_middleware",
    "without_middleware",
]


@pytest.fixture(params=middleware_modes)
def client(
    request: pytest.FixtureRequest,
    app: fastapi.FastAPI,
) -> collections.abc.Iterator[fastapi.testclient.TestClient]:
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
        with fastapi.testclient.TestClient(app) as c:
            yield c
    finally:
        # restore back the middlewares
        if request.param == "without_middleware":
            app.user_middleware = user_middleware
            app.middleware_stack = app.build_middleware_stack()


@pytest.fixture(autouse=True)
def _patch_k8s_service_discovery(monkeypatch):
    def _dummy_init(self, namespace=None, *_, **__):
        self.namespace = namespace

    async def _noop(*_, **__):
        return None

    monkeypatch.setattr(
        services.discovery.service.K8sServiceDiscovery,
        "__init__",
        _dummy_init,
        raising=False,
    )
    monkeypatch.setattr(
        services.discovery.service.K8sServiceDiscovery,
        "broadcast",
        _noop,
        raising=False,
    )


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
def test_set_and_get_log_level(
    log_config: dict[str, str],
    client: fastapi.testclient.TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MLRUN_NAMESPACE", "mlrun")

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
    "invalid_log_config, expected_status",
    [
        ({"mlrun_pipelines.imports": "INVALID_LEVEL"}, 422),  # bad value
        ({"mlrun_pipelines.imports": 123}, 422),  # schema type error
    ],
)
def test_invalid_log_config(
    invalid_log_config: dict[str, str],
    expected_status: int,
    client: fastapi.testclient.TestClient,
):
    payload = {"domain_to_levels": invalid_log_config, "recursive": False}
    response = client.post("/api/_internal/log_levels", json=payload)
    assert response.status_code == expected_status


def test_recursive_set_log_levels(
    client: fastapi.testclient.TestClient,
):
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


def test_non_recursive_preserves_children(
    client: fastapi.testclient.TestClient,
):
    domain = "mlrun"
    child_name = "mlrun.child"
    parent = logging.getLogger(domain)
    child = logging.getLogger(child_name)

    parent.setLevel(logging.NOTSET)
    child.setLevel(logging.WARNING)

    payload = {"domain_to_levels": {domain: "ERROR"}, "recursive": False}
    response = client.post("/api/_internal/log_levels", json=payload)
    assert response.status_code == 200

    assert parent.getEffectiveLevel() == logging.ERROR
    assert child.getEffectiveLevel() == logging.WARNING
