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
import http
import os

import fastapi.testclient
import pytest
import sqlalchemy.orm

from mlrun.utils import logger


def test_docs(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    response = client.get("openapi.json")
    assert response.status_code == http.HTTPStatus.OK.value


@pytest.mark.skipif(
    os.getenv("MLRUN_OPENAPI_JSON_NAME") is None,
    reason="Supposed to run only for CI backward compatibility tests",
)
def test_save_openapi_json(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    """The purpose of the test is to create an openapi.json file that is used to run backward compatibility tests"""
    response = client.get("openapi.json")
    path = os.path.abspath(os.getcwd())
    if os.getenv("MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH"):
        path = os.getenv("MLRUN_BC_TESTS_OPENAPI_OUTPUT_PATH")
    file_path = os.path.join(path, os.getenv("MLRUN_OPENAPI_JSON_NAME"))
    with open(file_path, "w") as file:
        file.write(response.text)
    logger.info("Saved openapi JSON file", file_path=file_path)
