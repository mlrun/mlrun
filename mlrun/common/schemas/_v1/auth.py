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


import pydantic.v1

from .._shared.auth import AuthInfoKind, AuthorizationAction, ProjectsRole


class AuthorizationVerificationInput(pydantic.v1.BaseModel):
    resource: str
    action: AuthorizationAction


class AuthInfo(pydantic.v1.BaseModel):
    # Keep request headers for inter-service communication
    request_headers: dict[str, str] | None = None
    # Basic + Iguazio auth
    username: str | None = None
    # Basic auth
    password: str | None = None
    # Bearer auth
    token: str | None = None
    # Iguazio auth
    session: str | None = None
    data_session: str | None = None
    access_key: str | None = None
    user_id: str | None = None
    user_group_ids: list[str] = []
    user_unix_id: int | None = None
    projects_role: ProjectsRole | None = None
    planes: list[str] = []
    kind: AuthInfoKind = AuthInfoKind.user

    def get_member_ids(self) -> list[str]:
        member_ids = []
        if self.user_id:
            member_ids.append(self.user_id)
        if self.username:
            member_ids.append(self.username)
        if self.user_group_ids:
            member_ids.extend(self.user_group_ids)
        return member_ids

    def get_session(self) -> str:
        return self.data_session or self.session

    def is_service_account(self) -> bool:
        return self.kind == AuthInfoKind.service_account


class Credentials(pydantic.v1.BaseModel):
    access_key: str | None


__all__ = [
    "AuthInfo",
    "AuthorizationVerificationInput",
    "Credentials",
]
