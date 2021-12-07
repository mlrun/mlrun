import enum
import typing

import pydantic
from nuclio.auth import AuthInfo as NuclioAuthInfo
from nuclio.auth import AuthKinds as NuclioAuthKinds


class ProjectsRole(str, enum.Enum):
    iguazio = "iguazio"
    mlrun = "mlrun"
    nuclio = "nuclio"
    nop = "nop"


class AuthorizationAction(str, enum.Enum):
    read = "read"
    create = "create"
    update = "update"
    delete = "delete"

    # note that in the OPA manifest only the above actions exist, store is "an MLRun verb" an we internally map it to 2
    # query permissions requests - create and update
    store = "store"


class AuthorizationResourceTypes(str, enum.Enum):
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
    marketplace_source = "marketplace-source"

    def to_resource_string(
        self, project_name: str, resource_name: str,
    ):
        return {
            # project is the resource itself, so no need for both resource_name and project_name
            AuthorizationResourceTypes.project: "/projects/{project_name}",
            AuthorizationResourceTypes.function: "/projects/{project_name}/functions/{resource_name}",
            AuthorizationResourceTypes.artifact: "/projects/{project_name}/artifacts/{resource_name}",
            # fmt: off
            AuthorizationResourceTypes.project_background_task:
                "/projects/{project_name}/background-tasks/{resource_name}",
            # fmt: on
            AuthorizationResourceTypes.background_task: "/background-tasks/{resource_name}",
            AuthorizationResourceTypes.feature_set: "/projects/{project_name}/feature-sets/{resource_name}",
            AuthorizationResourceTypes.feature_vector: "/projects/{project_name}/feature-vectors/{resource_name}",
            AuthorizationResourceTypes.feature: "/projects/{project_name}/features/{resource_name}",
            AuthorizationResourceTypes.entity: "/projects/{project_name}/entities/{resource_name}",
            AuthorizationResourceTypes.log: "/projects/{project_name}/runs/{resource_name}/logs",
            AuthorizationResourceTypes.schedule: "/projects/{project_name}/schedules/{resource_name}",
            AuthorizationResourceTypes.secret: "/projects/{project_name}/secrets/{resource_name}",
            AuthorizationResourceTypes.run: "/projects/{project_name}/runs/{resource_name}",
            # runtime resource doesn't have an identifier, we don't need any auth granularity behind project level
            AuthorizationResourceTypes.runtime_resource: "/projects/{project_name}/runtime-resources",
            AuthorizationResourceTypes.model_endpoint: "/projects/{project_name}/model-endpoints/{resource_name}",
            AuthorizationResourceTypes.pipeline: "/projects/{project_name}/pipelines/{resource_name}",
            # Marketplace sources are not project-scoped, and auth is globally on the sources endpoint.
            AuthorizationResourceTypes.marketplace_source: "/marketplace/sources",
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
    user_group_ids: typing.List[str] = []
    projects_role: typing.Optional[ProjectsRole] = None

    def to_nuclio_auth_info(self):
        if self.session != "":
            return NuclioAuthInfo(password=self.session, mode=NuclioAuthKinds.iguazio)
        return None

    def get_member_ids(self) -> typing.List[str]:
        member_ids = []
        if self.user_id:
            member_ids.append(self.user_id)
        if self.user_group_ids:
            member_ids.extend(self.user_group_ids)
        return member_ids


class Credentials(pydantic.BaseModel):
    access_key: typing.Optional[str]
