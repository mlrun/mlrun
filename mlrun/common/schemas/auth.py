# Copyright 2023 Iguazio
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
import typing

import pydantic
from nuclio.auth import AuthInfo as NuclioAuthInfo
from nuclio.auth import AuthKinds as NuclioAuthKinds

import mlrun.common.types


class ProjectsRole(mlrun.common.types.StrEnum):
    iguazio = "iguazio"
    mlrun = "mlrun"
    nuclio = "nuclio"
    nop = "nop"


class AuthorizationAction(mlrun.common.types.StrEnum):
    read = "read"
    create = "create"
    update = "update"
    delete = "delete"

    # note that in the OPA manifest only the above actions exist, store is "an MLRun verb" an we internally map it to 2
    # query permissions requests - create and update
    store = "store"


class AuthorizationResourceTypes(mlrun.common.types.StrEnum):
    project = "project"
    log = "log"
    runtime_resource = "runtime-resource"
    function = "function"
    artifact = "artifact"
    feature_set = "feature-set"
    feature_vector = "feature-vector"
    feature = "feature"
    entity = "entity"
    project_background_task = "project-background-task"
    background_task = "background-task"
    schedule = "schedule"
    secret = "secret"
    run = "run"
    model_endpoint = "model-endpoint"
    pipeline = "pipeline"
    hub_source = "hub-source"
    workflow = "workflow"
    alert = "alert"
    alert_templates = "alert-templates"
    event = "event"
    datastore_profile = "datastore-profile"
    api_gateway = "api-gateway"
    project_summaries = "project-summaries"

    def to_resource_string(
        self,
        project_name: str,
        resource_name: str,
    ):
        return {
            # project is the resource itself, so no need for both resource_name and project_name
            AuthorizationResourceTypes.project: "/projects/{project_name}",
            AuthorizationResourceTypes.project_summaries: "/projects/{project_name}/project-summaries/{resource_name}",
            AuthorizationResourceTypes.function: "/projects/{project_name}/functions/{resource_name}",
            AuthorizationResourceTypes.artifact: "/projects/{project_name}/artifacts/{resource_name}",
            AuthorizationResourceTypes.project_background_task: (
                "/projects/{project_name}/background-tasks/{resource_name}"
            ),
            AuthorizationResourceTypes.background_task: "/background-tasks/{resource_name}",
            AuthorizationResourceTypes.feature_set: "/projects/{project_name}/feature-sets/{resource_name}",
            AuthorizationResourceTypes.feature_vector: "/projects/{project_name}/feature-vectors/{resource_name}",
            AuthorizationResourceTypes.feature: "/projects/{project_name}/features/{resource_name}",
            AuthorizationResourceTypes.entity: "/projects/{project_name}/entities/{resource_name}",
            AuthorizationResourceTypes.log: "/projects/{project_name}/runs/{resource_name}/logs",
            AuthorizationResourceTypes.schedule: "/projects/{project_name}/schedules/{resource_name}",
            AuthorizationResourceTypes.secret: "/projects/{project_name}/secrets/{resource_name}",
            AuthorizationResourceTypes.run: "/projects/{project_name}/runs/{resource_name}",
            AuthorizationResourceTypes.event: "/projects/{project_name}/events/{resource_name}",
            AuthorizationResourceTypes.alert: "/projects/{project_name}/alerts/{resource_name}",
            AuthorizationResourceTypes.alert_templates: "/alert-templates/{resource_name}",
            # runtime resource doesn't have an identifier, we don't need any auth granularity behind project level
            AuthorizationResourceTypes.runtime_resource: "/projects/{project_name}/runtime-resources",
            AuthorizationResourceTypes.model_endpoint: "/projects/{project_name}/model-endpoints/{resource_name}",
            AuthorizationResourceTypes.pipeline: "/projects/{project_name}/pipelines/{resource_name}",
            AuthorizationResourceTypes.datastore_profile: "/projects/{project_name}/datastore_profiles",
            # Hub sources are not project-scoped, and auth is globally on the sources endpoint.
            # TODO - this was reverted to /marketplace since MLRun needs to be able to run with old igz versions. Once
            #  we only have support for igz versions that support /hub (>=3.5.4), change this to "/hub/sources".
            AuthorizationResourceTypes.hub_source: "/marketplace/sources",
            # workflow define how to run a pipeline and can be considered as the specification of a pipeline.
            AuthorizationResourceTypes.workflow: "/projects/{project_name}/workflows/{resource_name}",
            AuthorizationResourceTypes.api_gateway: "/projects/{project_name}/api-gateways/{resource_name}",
        }[self].format(project_name=project_name, resource_name=resource_name)


class AuthorizationVerificationInput(pydantic.BaseModel):
    resource: str
    action: AuthorizationAction


class AuthInfo(pydantic.BaseModel):
    # Basic + Iguazio auth
    username: typing.Optional[str] = None
    # Basic auth
    password: typing.Optional[str] = None
    # Bearer auth
    token: typing.Optional[str] = None
    # Iguazio auth
    session: typing.Optional[str] = None
    data_session: typing.Optional[str] = None
    access_key: typing.Optional[str] = None
    user_id: typing.Optional[str] = None
    user_group_ids: list[str] = []
    user_unix_id: typing.Optional[int] = None
    projects_role: typing.Optional[ProjectsRole] = None
    planes: list[str] = []

    def to_nuclio_auth_info(self):
        if self.session != "":
            return NuclioAuthInfo(password=self.session, mode=NuclioAuthKinds.iguazio)
        return None

    def get_member_ids(self) -> list[str]:
        member_ids = []
        if self.user_id:
            member_ids.append(self.user_id)
        if self.user_group_ids:
            member_ids.extend(self.user_group_ids)
        return member_ids


class Credentials(pydantic.BaseModel):
    access_key: typing.Optional[str]
