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
from typing import Optional

from mlrun.agentic.schemas.base import BaseWithVerMetadata
from mlrun.agentic.schemas.session import Conversation


class WorkflowType(str, Enum):
    INGESTION = "ingestion"
    APPLICATION = "application"
    DATA_PROCESSING = "data-processing"
    TRAINING = "training"
    EVALUATION = "evaluation"


class Workflow(BaseWithVerMetadata):
    _top_level_fields = ["workflow_type"]

    workflow_type: WorkflowType
    project_id: str
    deployment: Optional[str] = None
    workflow_function: Optional[str] = None
    configuration: Optional[dict] = None
    graph: Optional[dict] = None


class WorkflowEvent:
    """A workflow event."""

    def __init__(
        self,
        query=None,
        username=None,
        session_name=None,
        db_session=None,
        workflow_id=None,
        **kwargs,
    ):
        self.username = username
        self.session_name = session_name
        self.original_query = query
        self.query = query
        self.kwargs = kwargs

        self.session = None
        self.user = None
        self.results = {}
        self.state = {}
        self.conversation: Conversation = Conversation()
        self.workflow_id = workflow_id

        self.db_session = db_session

    def to_dict(self):
        return {
            "username": self.username,
            "session_id": self.session_id,
            "query": self.query,
            "kwargs": self.kwargs,
            "results": self.results,
            "state": self.state,
            "conversation": self.conversation.to_list(),
            "workflow_id": self.workflow_id,
            "session": self.session.to_dict() if self.session else None,
        }

    def __getitem__(self, item):
        return getattr(self, item)
