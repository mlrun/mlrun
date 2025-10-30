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

import pytest

import mlrun.common.schemas as schemas

import framework.utils.auth.verifier

TEST_PROJECT_NAME = "test-project"
TEST_RESOURCE_NAME = "test-resource"


@pytest.mark.parametrize(
    "resources_namespace, mgmt_namespace, resource_type, namespace, expected_resource_string",
    [
        (
            "resources",
            "mgmt",
            schemas.AuthorizationResourceTypes.function,
            schemas.AuthorizationResourceNamespace.resources,
            f"/resources/projects/{TEST_PROJECT_NAME}/functions/{TEST_RESOURCE_NAME}",
        ),
        (
            "resources",
            "mgmt",
            schemas.AuthorizationResourceTypes.project,
            schemas.AuthorizationResourceNamespace.resources,
            f"/resources/projects/{TEST_PROJECT_NAME}",
        ),
        (
            "resources",
            "mgmt",
            schemas.AuthorizationResourceTypes.function,
            schemas.AuthorizationResourceNamespace.mgmt,
            f"/mgmt/projects/{TEST_PROJECT_NAME}/functions/{TEST_RESOURCE_NAME}",
        ),
        (
            "res_ns",
            "mgmt",
            schemas.AuthorizationResourceTypes.project,
            schemas.AuthorizationResourceNamespace.mgmt,
            f"/mgmt/projects/{TEST_PROJECT_NAME}",
        ),
        (
            "",
            "",
            schemas.AuthorizationResourceTypes.function,
            schemas.AuthorizationResourceNamespace.resources,
            f"/projects/{TEST_PROJECT_NAME}/functions/{TEST_RESOURCE_NAME}",
        ),
        (
            "",
            "",
            schemas.AuthorizationResourceTypes.project,
            schemas.AuthorizationResourceNamespace.resources,
            f"/projects/{TEST_PROJECT_NAME}",
        ),
        (
            "",
            "",
            schemas.AuthorizationResourceTypes.function,
            schemas.AuthorizationResourceNamespace.mgmt,
            f"/projects/{TEST_PROJECT_NAME}/functions/{TEST_RESOURCE_NAME}",
        ),
        (
            "",
            "",
            schemas.AuthorizationResourceTypes.project,
            schemas.AuthorizationResourceNamespace.mgmt,
            f"/projects/{TEST_PROJECT_NAME}",
        ),
    ],
)
def test_attach_resource_namespace(
    resources_namespace: str,
    mgmt_namespace: str,
    resource_type: schemas.AuthorizationResourceTypes,
    namespace: schemas.AuthorizationResourceNamespace,
    expected_resource_string: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "mlrun.mlconf.httpdb.authorization.namespaces.resources", resources_namespace
    )
    monkeypatch.setattr(
        "mlrun.mlconf.httpdb.authorization.namespaces.mgmt", mgmt_namespace
    )
    resource_string = resource_type.to_resource_string(
        project_name=TEST_PROJECT_NAME, resource_name=TEST_RESOURCE_NAME
    )
    namespaced_resource_string = (
        framework.utils.auth.verifier.AuthVerifier()._attach_resource_namespace(
            resource_string, namespace
        )
    )
    assert namespaced_resource_string == expected_resource_string
