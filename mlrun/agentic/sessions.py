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

from mlrun.agentic.schemas import WorkflowEvent


class SessionStore:
    def __init__(self, client):
        self.db_session = None
        self.client = client

    def read_state(self, event: WorkflowEvent):
        event.user = self.client.get_user(
            username=event.username, email=event.username
        )
        event.username = event.user.name or "guest"
        if not event.session and event.session_name:
            event.session = self.client.get_session(
                name=event.session_name, username=event.username
            )
            event.conversation = event.session.to_conversation()

    def save(self, event: WorkflowEvent):
        if event.session_name:
            session = event.session
            session.history = event.conversation.to_list()
            self.client.update_session(
                chat_session=session,
                username=event.username,
            )
