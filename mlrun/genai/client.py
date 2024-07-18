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

import requests

from mlrun.genai.config import config, logger
from mlrun.utils.helpers import dict_to_json


class Client:
    def __init__(self, base_url, username=None, token=None):
        self.base_url = base_url
        self.username = username or "guest"
        self.token = token

    def post_request(
        self, path, data=None, params=None, method="GET", files=None, json=None
    ):
        # Construct the URL
        url = f"{self.base_url}/api/{path}"
        logger.debug(
            f"Sending {method} request to {url}, params: {params}, data: {data}"
        )
        kw = {
            key: value
            for key, value in (
                ("params", params),
                ("data", data),
                ("json", json),
                ("files", files),
            )
            if value is not None
        }
        if data is not None:
            kw["data"] = dict_to_json(kw["data"])
        if params is not None:
            kw["params"] = (
                {k: v for k, v in params.items() if v is not None} if params else None
            )
        # Make the request
        response = requests.request(
            method,
            url,
            headers={"x_username": self.username},
            **kw,
        )

        # Check the response
        if response.status_code == 200:
            # If the request was successful, return the JSON response
            return response.json()
        else:
            # If the request failed, raise an exception
            response.raise_for_status()

    def get_collection(self, name):
        response = self.post_request(f"collection/{name}")
        return response["data"]

    def get_session(self, session_id):
        response = self.post_request(f"session/{session_id}")
        return response

    def get_user(self, username):
        response = self.post_request(f"user/{username}")
        return response["data"]

    def create_session(
        self,
        name,
        username=None,
        agent_name=None,
        history=None,
        features=None,
        state=None,
    ):
        chat_session = {
            "name": name,
            "username": username,
            "agent_name": agent_name,
            "history": history,
            "features": features,
            "state": state,
        }
        response = self.post_request("session", data=chat_session, method="POST")
        return response["success"]

    def update_session(
        self,
        name,
        username=None,
        agent_name=None,
        history=None,
        features=None,
        state=None,
    ):
        chat_session = {
            "name": name,
            "username": username or self.username,
            "agent_name": agent_name,
            "history": history,
            "features": features,
            "state": state,
        }
        response = self.post_request(f"session/{name}", data=chat_session, method="PUT")
        return response["success"]


client = Client(base_url=config.api_url)
