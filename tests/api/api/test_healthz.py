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

import fastapi.testclient
import sqlalchemy.orm

import mlrun
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.errors
import mlrun.runtimes


def test_health(
    db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
) -> None:
    overridden_ui_projects_prefix = "some-prefix"
    mlrun.mlconf.ui.projects_prefix = overridden_ui_projects_prefix
    nuclio_version = "x.x.x"
    mlrun.mlconf.nuclio_version = nuclio_version
    response = client.get("healthz")
    assert response.status_code == http.HTTPStatus.OK.value
    response_body = response.json()
    for key in ["scrape_metrics", "hub_url"]:
        assert response_body[key] is None
    assert response_body["ui_projects_prefix"] == overridden_ui_projects_prefix
    assert response_body["nuclio_version"] == nuclio_version
