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

from mlrun.genai.chains.base import ChainRunner
from mlrun.genai.config import get_llm, logger
from mlrun.genai.schema import PipelineEvent

_refine_prompt_template = """
You are a helpful AI assistant, given the following conversation and a follow up request,
 rephrase the follow up request to be a standalone request, keeping the same user language.
Your rephrasing must include any relevant history element to get a precise standalone request
 and not losing previous context.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone request:
"""


class RefineQuery(ChainRunner):
    def __init__(self, llm=None, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template
        self._chain = None

    def post_init(self, mode="sync"):
        self.llm = self.llm or get_llm(self.context._config)
        refine_prompt = PromptTemplate.from_template(
            self.prompt_template or _refine_prompt_template
        )
        self._chain = refine_prompt | self.llm

    def _run(self, event: PipelineEvent):
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
