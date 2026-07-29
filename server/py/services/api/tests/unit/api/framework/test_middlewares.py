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

import json
import unittest.mock
import uuid
from http import HTTPStatus

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.common.schemas.constants
import mlrun.utils.version

import framework.middlewares


@pytest.mark.parametrize(
    "ui_version,backend_version,clear_cache",
    [
        # ui version was not sent, no need to clear cache
        ("", "0.0.1", False),
        # matching version, no need to clear cache
        ("1.0.0", "1.0.0", False),
        # development version, no need to clear cache
        ("0.0.1", "0.0.0", False),
        # non-matching version, need to clear cache
        ("0.0.1", "0.0.2", True),
        ("0.0.0", "0.0.1", True),
        ("0.0.2", "0.0.1", True),
    ],
)
def test_ui_clear_cache_middleware(
    db: sqlalchemy.orm.Session,
    client: fastapi.testclient.TestClient,
    ui_version: str,
    backend_version: str,
    clear_cache: bool,
) -> None:
    for middleware in client.app.user_middleware:
        if "UiClearCacheMiddleware" in str(middleware.cls):
            middleware.kwargs["backend_version"] = backend_version
    client.app.middleware_stack = client.app.build_middleware_stack()

    with unittest.mock.patch.object(
        mlrun.utils.version.Version, "get", return_value={"version": backend_version}
    ):
        response = client.get(
            "client-spec",
            headers={
                mlrun.common.schemas.constants.HeaderNames.ui_version: ui_version,
            },
        )

    if clear_cache:
        assert response.headers["Clear-Site-Data"] == '"cache"'
        assert (
            response.headers[mlrun.common.schemas.constants.HeaderNames.ui_clear_cache]
            == "true"
        )
    else:
        assert "Clear-Site-Data" not in response.headers
        assert (
            mlrun.common.schemas.constants.HeaderNames.ui_clear_cache
            not in response.headers
        )


def test_ensure_be_version_middleware(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    for middleware in client.app.user_middleware:
        if "backend_version" in middleware.kwargs:
            middleware.kwargs["backend_version"] = "dummy-version"
    client.app.middleware_stack = client.app.build_middleware_stack()
    response = client.get("client-spec")
    assert (
        response.headers[mlrun.common.schemas.constants.HeaderNames.backend_version]
        == "dummy-version"
    )


async def _noop_receive():
    return {"type": "http.disconnect"}


async def _noop_send(message):
    pass


async def _run_ensure_json_content_type_middleware(scope: dict) -> dict:
    downstream_scope = {}

    async def downstream_app(inner_scope, receive, send):
        downstream_scope.update(inner_scope)

    middleware = framework.middlewares.EnsureJsonContentTypeMiddleware(downstream_app)
    await middleware(scope, _noop_receive, _noop_send)
    return downstream_scope


async def test_ensure_json_content_type_middleware_defaults_missing_header() -> None:
    downstream_scope = await _run_ensure_json_content_type_middleware(
        {"type": "http", "headers": []}
    )
    assert dict(downstream_scope["headers"])[b"content-type"] == b"application/json"


async def test_ensure_json_content_type_middleware_does_not_override_existing_content_type() -> (
    None
):
    downstream_scope = await _run_ensure_json_content_type_middleware(
        {
            "type": "http",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=some-boundary")
            ],
        }
    )
    assert (
        dict(downstream_scope["headers"])[b"content-type"]
        == b"multipart/form-data; boundary=some-boundary"
    )


async def test_ensure_json_content_type_middleware_skips_non_http_scope() -> None:
    downstream_scope = await _run_ensure_json_content_type_middleware(
        {"type": "websocket", "headers": []}
    )
    assert downstream_scope["headers"] == []


def test_ensure_json_content_type_middleware_lets_headerless_json_body_through(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    # simulates a v1-client sending a pre-serialized JSON body without a Content-Type header
    # (e.g. mlrun.db.httpdb.HTTPRunDB.api_call's `body=` kwarg, which requests/httpx never tag
    # with Content-Type on their own, unlike the `json=` kwarg)
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name=f"prj-{uuid.uuid4().hex}"),
        spec=mlrun.common.schemas.ProjectSpec(),
    )
    response = client.post("projects", content=json.dumps(project.dict()).encode())
    assert response.status_code == HTTPStatus.CREATED.value
