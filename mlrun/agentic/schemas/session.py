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

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel

from mlrun.agentic.schemas.base import BaseWithOwner


class QueryItem(BaseModel):
    question: str
    session_name: Optional[str] = None
    filter: Optional[List[Tuple[str, str]]] = None
    data_source: Optional[str] = None


class ChatRole(str, Enum):
    HUMAN = "Human"
    AI = "AI"
    SYSTEM = "System"
    USER = "User"
    AGENT = "Agent"


class Message(BaseModel):
    role: ChatRole
    content: str
    extra_data: Optional[dict] = None
    sources: Optional[List[dict]] = None
    human_feedback: Optional[str] = None


class Conversation(BaseModel):
    messages: list[Message] = []
    saved_index: int = 0

    def __str__(self):
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])

    def add_message(self, role, content, sources=None):
        self.messages.append(Message(role=role, content=content, sources=sources))

    def to_list(self):
        return self.model_dump(mode="json")["messages"]

    def to_dict(self):
        return self.model_dump(mode="json")["messages"]

    @classmethod
    def from_list(cls, data: list):
        return cls.model_validate({"messages": data or []})


class ChatSession(BaseWithOwner):
    _extra_fields = ["history"]
    _top_level_fields = ["workflow_id"]

    workflow_id: str
    history: List[Message] = []

    def to_conversation(self):
        return Conversation.from_list(self.history)
