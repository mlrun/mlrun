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
    log = "log"
    runtime_resource = "runtime-resource"
    function = "function"
    artifact = "artifact"
    feature_set = "feature-set"
    feature_vector = "feature-vector"
    feature = "feature"
    entity = "entity"
    background_task = "background-task"
    schedule = "schedule"
    secret = "secret"
    run = "run"
    model_endpoint = "model-endpoint"


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
    user_id: typing.Optional[str] = None
    user_group_ids: typing.List[str] = []
    projects_role: typing.Optional[ProjectsRole] = None

    def to_nuclio_auth_info(self):
        if self.session != "":
            return NuclioAuthInfo(password=self.session, mode=NuclioAuthKinds.iguazio)
        return None
