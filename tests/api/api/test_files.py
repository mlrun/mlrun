from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# fixtures for test, aren't used directly so we need to ignore the lint here
from tests.common_fixtures import (  # noqa: F401
    patch_file_forbidden,
    patch_file_not_found,
)


@pytest.mark.usefixtures("patch_file_forbidden")
def test_files_forbidden(db: Session, client: TestClient) -> None:
    validate_files_status_code(client, HTTPStatus.FORBIDDEN.value)


@pytest.mark.usefixtures("patch_file_not_found")
def test_files_not_found(db: Session, client: TestClient) -> None:
    validate_files_status_code(client, HTTPStatus.NOT_FOUND.value)


def validate_files_status_code(client: TestClient, status_code: int):
    resp = client.get("files?schema=v3io&path=mybucket/files.txt")
    assert resp.status_code == status_code

    resp = client.get("files?schema=v3io&path=mybucket/")
    assert resp.status_code == status_code

    resp = client.get("filestat?schema=v3io&path=mybucket/files.txt")
    assert resp.status_code == status_code
