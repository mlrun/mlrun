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

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent

COMMUNICATOR_PROMPT = """
You are a professional AI Workspace Assistant.

Instructions:
- Maintain a professional, respectful, and helpful tone at all times.
- Avoid any profanity or offensive language.

User message:
{query}
"""


class Communicator(ChainRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = None
        self._chain = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        return self._llm

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        prompt = PromptTemplate.from_template(COMMUNICATOR_PROMPT)
        self._chain = prompt | self.llm

    def _run(self, event: WorkflowEvent):
        response = self._chain.invoke({"query": event.query})
        return {"answer": response.content}
