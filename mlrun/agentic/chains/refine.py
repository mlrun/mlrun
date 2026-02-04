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

from langchain_core.prompts.prompt import PromptTemplate

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.config import get_llm
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.agentic.utils import logger

_refine_prompt_template = """
You are an assistant refining a user query for retrieval.
You have full access to the provided chat history in this conversation.
Use it when necessary to clarify ambiguous references in the current query.

Rules:
- The current user query always takes priority over chat history.
- Use chat history ONLY to clarify ambiguous references.
- If the user query is a greeting or small talk, leave it as conversational intent.

Input:
Chat History: {chat_history}
Current User Query: {question}

output:
"""

CONVERSATION_CONTEXT_REFINER_PROMPT = """
You are a conversation context refiner.

Your job is to prepare the best possible input for downstream reasoning while preserving the user's original intent.

CRITICAL RULES:
1. DO NOT summarize unless explicitly instructed
2. DO NOT invent goals, tasks, or instructions
3. DO NOT change the meaning or intent of the user's input
4. Output only the refined input — no explanations

For regular chat messages: Use chat history ONLY to clarify ambiguous references.
For meeting transcripts: Return the ENTIRE transcript exactly as provided.

INPUTS:
Chat History:
{chat_history}

Current Input:
{question}

OUTPUT:
Return ONLY the refined input text. No explanations.
"""


class RefineQuery(ChainRunner):
    def __init__(self, llm=None, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template
        self._chain = None

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        self.llm = self.llm or get_llm(self.context._config)
        refine_prompt = PromptTemplate.from_template(
            self.prompt_template or _refine_prompt_template
        )
        self._chain = refine_prompt | self.llm

    def _run(self, event: WorkflowEvent):
        chat_history = str(event.conversation)
        logger.debug(f"Question: {event.query}\nChat history: {chat_history}")
        resp = self._chain.invoke(
            {"question": event.query, "chat_history": chat_history}
        )
        logger.debug(f"Refined question: {resp}")
        return {"answer": resp}


def get_refine_chain(config, verbose=False, prompt_template=None):
    llm = get_llm(config)
    verbose = verbose or config.verbose
    return RefineQuery(llm=llm, verbose=verbose, prompt_template=prompt_template)
