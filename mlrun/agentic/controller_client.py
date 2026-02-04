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

from typing import Union

import requests

import mlrun.utils.helpers
from mlrun.agentic.schemas import ChatSession, DataSource, Project, User, Workflow
from mlrun.agentic.utils import logger


class ControllerClient:
    def __init__(
        self, controller_url: str, project_name: str, username: str, token: str = None
    ):
        self._controller_url = controller_url
        self._project_name = project_name
        self._username = username
        self._token = token
        self._project_id = None
        self._owner_id = None

    @property
    def project_id(self):
        if not self._project_id:
            self._project_id = self.get_project().uid
        return self._project_id

    @property
    def owner_id(self):
        if not self._owner_id:
            self._owner_id = self.get_user().uid
        return self._owner_id

    def _send_request(
        self,
        path: str,
        method: str,
        params: dict = None,
        data: dict = None,
        files: dict = None,
        json: dict = None,
    ):
        url = f"{self._controller_url}/api/{path}"

        request_kwargs = {}
        if data is not None:
            request_kwargs["data"] = mlrun.utils.helpers.dict_to_json(data)
        if params is not None:
            request_kwargs["params"] = {
                k: v for k, v in params.items() if v is not None
            }
        if json is not None:
            request_kwargs["json"] = json
        if files is not None:
            request_kwargs["files"] = files

        logger.debug(
            f"Sending {method} request to {url}, params: {params}, data: {data}"
        )
        response = requests.request(
            method,
            url,
            headers={"x_username": self._username},
            **request_kwargs,
        )

        if response.status_code == 200:
            return response.json()
        response.raise_for_status()

    def get_data_source(
        self, name: str, uid: str = None, version: str = None
    ) -> DataSource:
        params = {}
        if uid:
            params["uid"] = uid
        if version:
            params["version"] = version
        response = self._send_request(
            path=f"projects/{self._project_name}/data_sources/{name}",
            method="GET",
            params=params,
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return DataSource(**dict_response)

    def get_session(
        self, name: str, uid: str = None, username: str = None
    ) -> ChatSession:
        username = username or self._username
        params = {}
        if uid:
            params["uid"] = uid
        response = self._send_request(
            path=f"users/{username}/sessions/{name}", method="GET", params=params
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return ChatSession(**dict_response)

    def get_user(self, username: str = "", email: str = None, uid: str = None) -> User:
        username = username or self._username
        params = {}
        if email:
            params["email"] = email
        if uid:
            params["uid"] = uid
        response = self._send_request(
            path=f"users/{username}", method="GET", params=params
        )
        raw_response = response["data"]
        user_data = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return User(**user_data)

    def update_session(
        self,
        chat_session: ChatSession,
        username: str = None,
    ) -> ChatSession:
        username = username or self._username
        response = self._send_request(
            path=f"users/{username}/sessions/{chat_session.name}",
            method="PUT",
            data=chat_session.to_dict(),
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return ChatSession(**dict_response)

    def get_project(self) -> Project:
        response = self._send_request(
            path=f"projects/{self._project_name}", method="GET"
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return Project(**dict_response)

    def create_workflow(self, workflow: Union[Workflow, dict]) -> Workflow:
        project_id = self.get_project().uid
        if isinstance(workflow, dict):
            workflow["graph"] = [step.to_dict() for step in workflow["graph"]]
            workflow = Workflow(**workflow)

        workflow.project_id = project_id
        response = self._send_request(
            path=f"projects/{self._project_name}/workflows",
            method="POST",
            data=workflow.to_dict(),
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return Workflow(**dict_response)

    def get_workflow(
        self, workflow_name: str = None, workflow_id: str = None, version: str = None
    ) -> Workflow:
        params = {}
        if workflow_id:
            params["uid"] = workflow_id
        if version:
            params["version"] = version
        response = self._send_request(
            path=f"projects/{self._project_name}/workflows/{workflow_name}",
            method="GET",
            params=params,
        )
        return Workflow(**response)

    def update_workflow(self, workflow: Workflow) -> Workflow:
        response = self._send_request(
            path=f"projects/{self._project_name}/workflows/{workflow.name}",
            method="PUT",
            data=workflow.to_dict(),
        )
        raw_response = response["data"]
        dict_response = (
            dict(raw_response) if isinstance(raw_response, list) else raw_response
        )
        return Workflow(**dict_response)
