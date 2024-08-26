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

from mlrun.genai.client import Client
from mlrun.genai.client import client as default_client
from mlrun.genai.schemas import ChatSession, WorkflowEvent


class SessionStore:
    def __init__(self, client):
        self.db_session = None
        self.client = client

    def read_state(self, event: WorkflowEvent):
        event.user = self.client.get_user(username=event.username, email=event.username)
        event.username = event.user["name"] or "guest"
        if not event.session and event.session_id:
            resp = self.client.get_session(
                uid=event.session_id, user_name=event.username
            )
            chat_session = ChatSession(**resp)
            event.session = chat_session
            event.conversation = chat_session.to_conversation()

    def save(self, event: WorkflowEvent):
        """Save the session and conversation to the database"""
        if event.session_id:
            self.client.update_session(
                chat_session=event.session,
                username=event.username,
                history=event.conversation.to_list(),
            )


def get_session_store(config=None):
    if config:
        client = Client(base_url=config.api_url)
    else:
        client = default_client
    return SessionStore(client=client)
