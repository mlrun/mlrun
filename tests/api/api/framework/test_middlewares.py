import unittest.mock

import fastapi.testclient
import pytest
import sqlalchemy.orm

import mlrun.api.schemas.constants
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
    with unittest.mock.patch.object(
        mlrun.utils.version.Version, "get", return_value={"version": backend_version}
    ):
        response = client.get(
            f"client-spec",
            headers={
                mlrun.api.schemas.constants.HeaderNames.ui_version: ui_version,
            },
        )

    if clear_cache:
        assert response.headers["Clear-Site-Data"] == '"cache"'
        assert (
            response.headers[mlrun.api.schemas.constants.HeaderNames.ui_clear_cache]
            == "true"
        )
    else:
        assert "Clear-Site-Data" not in response.headers
        assert (
            mlrun.api.schemas.constants.HeaderNames.ui_clear_cache
            not in response.headers
        )


def test_ensure_be_version_middleware(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    with unittest.mock.patch.object(
        mlrun.utils.version.Version, "get", return_value={"version": "dummy-version"}
    ) as mock_version_get:
        response = client.get("client-spec")
        assert (
            response.headers[mlrun.api.schemas.constants.HeaderNames.backend_version]
            == mock_version_get.return_value["version"]
        )
