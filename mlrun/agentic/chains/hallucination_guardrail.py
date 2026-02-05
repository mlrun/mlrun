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

HALLUCINATION_GUARDRAIL_PROMPT = """
You are a strict factual consistency checker.

Source text:
---
{source}
---

Generated answer:
---
{answer}
---

Is the generated answer fully supported by the source text?

Respond with ONLY one word: SUPPORTED or UNSUPPORTED
"""


class HallucinationGuardrail(ChainRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = None
        self._chain = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return self._llm

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        prompt = PromptTemplate.from_template(HALLUCINATION_GUARDRAIL_PROMPT)
        self._chain = prompt | self.llm

    def _run(self, event: WorkflowEvent):
        source = event.query
        answer = event.results.get("answer", "")

        if not source or not answer:
            return {}

        verdict = self._chain.invoke(
            {"source": source, "answer": answer}
        ).content.strip()

        if verdict != "SUPPORTED":
            return {
                "stop": True,
                "error_message": "The response is not grounded in the provided information.",
            }

        return {"hallucination guardrail": "valid"}
