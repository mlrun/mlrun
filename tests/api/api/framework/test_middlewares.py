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

import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.common.schemas.constants
import mlrun.utils.version


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
