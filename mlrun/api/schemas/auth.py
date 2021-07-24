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
