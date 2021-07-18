import enum
import typing

import pydantic


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
