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
import logging

import requests
import requests.adapters
import urllib3.util

import mlrun.common.schemas
import mlrun.utils
import mlrun.utils.http
from mlrun.utils.logger import context_id_var

# Direct calls to Orca's API: one attempt per request, retried only when the connection could not be
# established, so nothing reached Orca - revoking a token and publishing an event are not idempotent.
ORCA_REQUEST_TIMEOUT_SECONDS = 60
ORCA_CONNECT_RETRIES = 3


def new_orca_session() -> requests.Session:
    """
    Create a session for direct calls to Orca's API. Every request on it must pass
    ``timeout=ORCA_REQUEST_TIMEOUT_SECONDS``, ``verify=mlrun.mlconf.iguazio_api_ssl_verify`` and
    ``allow_redirects=False``. ``verify`` goes on the request rather than the session because
    requests lets ``REQUESTS_CA_BUNDLE`` override a session-level ``verify=False``.

    The session never stores cookies: it is shared by calls made for different identities, so a
    cookie from one response must never be sent on a later request.
    """
    session = requests.Session()
    session.cookies = mlrun.utils.http.DummyCookieJar()
    retry = urllib3.util.Retry(
        total=ORCA_CONNECT_RETRIES,
        connect=ORCA_CONNECT_RETRIES,
        read=0,
        status=0,
        other=0,
        backoff_factor=0.5,
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def iguazio_sdk_logger(parent_logger: mlrun.utils.Logger) -> logging.Logger:
    """
    Return the stdlib logger to hand the Iguazio SDK so MLRun owns its logging.

    Without a logger the SDK installs its own stdout handler and also propagates into
    MLRun's logger, duplicating every line. A dedicated "sdk" child is used so the SDK's
    internal set_level("INFO") doesn't raise MLRun's own logger.
    """
    return parent_logger.get_child("sdk")._logger


def enrich_headers(headers: dict | None = None, path: str | None = None) -> dict:
    """
    Enrich headers with context id for logging correlation and project role header for project paths
    """
    headers = headers or {}

    inject_context_id_header(headers)

    if (
        path is not None
        and "projects" in path
        and mlrun.common.schemas.HeaderNames.projects_role not in headers
    ):
        headers[mlrun.common.schemas.HeaderNames.projects_role] = "mlrun"

    return headers


def inject_context_id_header(headers: dict):
    if mlrun.common.schemas.HeaderNames.igz_ctx not in headers:
        if (ctx_id := context_id_var.get()) is not None:
            headers[mlrun.common.schemas.HeaderNames.igz_ctx] = ctx_id
