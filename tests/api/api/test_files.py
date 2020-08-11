from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.common_fixtures import patch_file_forbidden, patch_file_not_found


def test_files_forbidden(patch_file_forbidden, db: Session, client: TestClient) -> None:
    validate_files_status_code(client, HTTPStatus.FORBIDDEN.value)


def test_files_not_found(patch_file_not_found, db: Session, client: TestClient) -> None:
    validate_files_status_code(client, HTTPStatus.NOT_FOUND.value)


def validate_files_status_code(client: TestClient, status_code: int):
    resp = client.get('/api/files?schema=v3io&path=mybucket/files.txt')
    assert resp.status_code == status_code

    resp = client.get('/api/files?schema=v3io&path=mybucket/')
    assert resp.status_code == status_code

    resp = client.get('/api/filestat?schema=v3io&path=mybucket/files.txt')
    assert resp.status_code == status_code