# Copyright 2026 Iguazio
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

import pytest

import mlrun
import mlrun.common.schemas
from mlrun.utils.logger import context_id_var

import framework.utils.clients.helpers as clients_helpers


@pytest.fixture()
def set_context_id():
    """
    Fixture to set the context ID variable and reset it after the test.
    Returns a function that can be used to set the context ID variable.
    """
    tokens = []

    def _set_context_id(context_id):
        token = context_id_var.set(context_id)
        tokens.append(token)
        return context_id

    yield _set_context_id

    # Cleanup: reset all tokens in reverse order
    for token in reversed(tokens):
        context_id_var.reset(token)


def test_inject_context_id_header_adds_header_when_context_set(set_context_id):
    context_id = "test-context-id-12345"
    headers = {}

    # Set the context variable
    set_context_id(context_id)
    clients_helpers.inject_context_id_header(headers)
    assert mlrun.common.schemas.HeaderNames.igz_ctx in headers
    assert headers[mlrun.common.schemas.HeaderNames.igz_ctx] == context_id


def test_inject_context_id_header_does_not_override_existing(set_context_id):
    existing_context_id = "existing-context-id"
    new_context_id = "new-context-id"
    headers = {mlrun.common.schemas.HeaderNames.igz_ctx: existing_context_id}

    # Set a different context variable
    set_context_id(new_context_id)
    clients_helpers.inject_context_id_header(headers)
    # Should keep the existing header value
    assert headers[mlrun.common.schemas.HeaderNames.igz_ctx] == existing_context_id


def test_inject_context_id_header_no_op_when_no_context(set_context_id):
    headers = {}

    # Ensure context_id_var is None
    set_context_id(None)
    clients_helpers.inject_context_id_header(headers)
    assert mlrun.common.schemas.HeaderNames.igz_ctx not in headers


def test_inject_context_id_header_with_other_headers_present(set_context_id):
    context_id = "test-context-id"
    headers = {
        "Authorization": "Bearer token123",
        "Content-Type": "application/json",
    }

    set_context_id(context_id)
    clients_helpers.inject_context_id_header(headers)
    # Should add context id header
    assert headers[mlrun.common.schemas.HeaderNames.igz_ctx] == context_id
    # Should preserve existing headers
    assert headers["Authorization"] == "Bearer token123"
    assert headers["Content-Type"] == "application/json"


# Tests for enrich_headers function


def test_enrich_headers_creates_dict_when_headers_is_none(set_context_id):
    set_context_id(None)  # No context ID to avoid side effects

    result = clients_helpers.enrich_headers(headers=None)

    assert result == {}
    assert isinstance(result, dict)


def test_enrich_headers_injects_context_id(set_context_id):
    context_id = "enriched-context-id"
    set_context_id(context_id)

    result = clients_helpers.enrich_headers(headers={})

    assert result[mlrun.common.schemas.HeaderNames.igz_ctx] == context_id


def test_enrich_headers_adds_projects_role(set_context_id):
    """
    Test that the projects role header is always added when the path contains "projects"
    """
    set_context_id(None)
    result = clients_helpers.enrich_headers(
        headers={}, path="/api/v1/projects/my-project"
    )
    assert (
        result[mlrun.common.schemas.HeaderNames.projects_role]
        == mlrun.mlconf.httpdb.projects.leader
    )


def test_enrich_headers_does_not_add_projects_role_when_path_is_none(set_context_id):
    set_context_id(None)
    result = clients_helpers.enrich_headers(headers={}, path=None)
    assert mlrun.common.schemas.HeaderNames.projects_role not in result


def test_enrich_headers_does_not_add_projects_role_when_path_has_no_projects(
    set_context_id,
):
    set_context_id(None)
    result = clients_helpers.enrich_headers(
        headers={}, path="/api/v1/functions/my-function"
    )
    assert mlrun.common.schemas.HeaderNames.projects_role not in result


def test_enrich_headers_does_not_override_existing_projects_role(set_context_id):
    set_context_id(None)
    existing_role = "custom-role"

    result = clients_helpers.enrich_headers(
        headers={mlrun.common.schemas.HeaderNames.projects_role: existing_role},
        path="/api/v1/projects/my-project",
    )
    assert result[mlrun.common.schemas.HeaderNames.projects_role] == existing_role


def test_enrich_headers_preserves_existing_headers(set_context_id):
    context_id = "test-context"
    set_context_id(context_id)

    headers = {
        "Authorization": "Bearer token",
        "X-Custom-Header": "custom-value",
    }
    result = clients_helpers.enrich_headers(
        headers=headers, path="/api/v1/projects/my-project"
    )

    # Should preserve existing headers
    assert result["Authorization"] == "Bearer token"
    assert result["X-Custom-Header"] == "custom-value"
    # Should add context id
    assert result[mlrun.common.schemas.HeaderNames.igz_ctx] == context_id
    # Should add projects role
    assert (
        result[mlrun.common.schemas.HeaderNames.projects_role]
        == mlrun.mlconf.httpdb.projects.leader
    )
