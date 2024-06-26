from enum import Enum
from http.client import HTTPException
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class ChatRole(str, Enum):
    Human = "Human"
    AI = "AI"
    System = "System"
    User = "User"  # for co-pilot user (vs Human?)
    Agent = "Agent"  # for co-pilot agent


class Message(BaseModel):
    role: ChatRole
    content: str
    html: Optional[str] = None
    sources: Optional[List[dict]] = None
    rating: Optional[int] = None
    suggestion: Optional[str] = None


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


class PipelineEvent:
    """A pipeline event."""

    def __init__(
        self, query=None, username=None, session_id=None, db_session=None, **kwargs
    ):
        self.username = username
        self.session_id = session_id
        self.original_query = query
        self.query = query
        self.kwargs = kwargs

        self.session = None
        self.user = None
        self.results = {}
        self.state = {}
        self.conversation: Conversation = Conversation()

        self.db_session = db_session  # SQL db session (from FastAPI)

    def to_dict(self):
        return {
            "username": self.username,
            "session_id": self.session_id,
            "query": self.query,
            "kwargs": self.kwargs,
            "results": self.results,
            "state": self.state,
            "conversation": self.conversation.to_list(),
        }

    def __getitem__(self, item):
        return getattr(self, item)


class TerminateResponse(BaseModel):
    success: bool = True
    error: Optional[str] = None
    resp: dict = {}

    def with_raise(self, format=None) -> "TerminateResponse":
        if not self.success:
            format = format or "Pipeline step failed: %s"
            raise ValueError(format % self.error)
        return self


class QueryItem(BaseModel):
    question: str
    session_id: Optional[str] = None
    filter: Optional[List[Tuple[str, str]]] = None
    collection: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Union[list, BaseModel, dict]] = None
    error: Optional[str] = None

    def with_raise(self, format=None) -> "ApiResponse":
        if not self.success:
            format = format or "API call failed: %s"
            raise ValueError(format % self.error)
        return self

    def with_raise_http(self, format=None) -> "ApiResponse":
        if not self.success:
            format = format or "API call failed: %s"
            raise HTTPException(status_code=400, detail=format % self.error)
        return self


class ApiDictResponse(ApiResponse):
    data: Optional[dict] = None


class PromptConfig(BaseModel):
    name: str
    description: Optional[str] = None
    labels: Optional[Dict[str, Union[str, None]]] = None
    template: str
    inputs: dict = None
    llm_args: dict = None
