# Copyright 2025 Iguazio
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
import mlrun.common.schemas
from mlrun.utils.logger import context_id_var


def enrich_headers(headers: dict | None = None, path: str | None = None) -> dict:
    """
    Enrich headers with context id for logging correlation and project role header for project paths
    """
    headers = headers or {}

    inject_context_id_header(headers)

    if (
        mlrun.mlconf.httpdb.projects.leader == "mlrun"
        and path is not None
        and "projects" in path
        and mlrun.common.schemas.HeaderNames.projects_role not in headers
    ):
        headers[mlrun.common.schemas.HeaderNames.projects_role] = "mlrun"

    return headers


def inject_context_id_header(headers: dict):
    if mlrun.common.schemas.HeaderNames.igz_ctx not in headers:
        if (ctx_id := context_id_var.get()) is not None:
            headers[mlrun.common.schemas.HeaderNames.igz_ctx] = ctx_id
