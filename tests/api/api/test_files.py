# Copyright 2018 Iguazio
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
