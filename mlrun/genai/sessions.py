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
from mlrun.genai.schema import ChatSession, PipelineEvent


class SessionStore:
    def __init__(self, client):
        self.db_session = None
        self.client = client

    def read_state(self, event: PipelineEvent):
        event.username = event.username or "guest"
        event.user = self.client.get_user(event.username)
        if event.session_id:
            resp = self.client.get_session(event.session_id)["data"]
            if resp:
                chat_session = ChatSession(**resp)
                event.session = chat_session
                event.state = chat_session.state
                event.conversation = chat_session.to_conversation()
            else:
                self.client.create_session(
                    name=event.session_id,
                    username=event.username or "guest",
                )

    def save(self, event: PipelineEvent):
        """Save the session and conversation to the database"""
        if event.session_id:
            self.client.update_session(
                id=event.session_id,
                # state=event.state,
                history=event.conversation.to_list(),
            )


def get_session_store(config=None):
    if config:
        client = Client(base_url=config.api_url)
    else:
        client = default_client
    return SessionStore(client=client)
