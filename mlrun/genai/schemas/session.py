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

from mlrun.genai.schemas.base import BaseWithOwner


class QueryItem(BaseModel):
    question: str
    session_id: Optional[str]
    filter: Optional[List[Tuple[str, str]]]
    data_source: Optional[str]


class ChatRole(str, Enum):
    HUMAN = "Human"
    AI = "AI"
    SYSTEM = "System"
    USER = "User"  # for co-pilot user (vs Human?)
    AGENT = "Agent"  # for co-pilot agent


class Message(BaseModel):
    role: ChatRole
    content: str
    extra_data: Optional[dict]
    sources: Optional[List[dict]]
    human_feedback: Optional[str]


class Conversation(BaseModel):
    messages: list[Message] = []
    saved_index: int = 0

    def __str__(self):
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])

    def add_message(self, role, content, sources=None):
        self.messages.append(Message(role=role, content=content, sources=sources))

    def to_list(self):
        return self.dict()["messages"]
        # return self.model_dump(mode="json")["messages"]

    def to_dict(self):
        return self.dict()["messages"]
        # return self.model_dump(mode="json")["messages"]

    @classmethod
    def from_list(cls, data: list):
        return cls.parse_obj({"messages": data or []})
        # return cls.model_validate({"messages": data or []})


class ChatSession(BaseWithOwner):
    _extra_fields = ["history"]
    _top_level_fields = ["workflow_id"]

    workflow_id: str
    history: Optional[List[Message]] = []

    def to_conversation(self):
        return Conversation.from_list(self.history)
