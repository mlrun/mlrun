from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ..config import get_llm, logger
from ..schema import PipelineEvent
from .base import ChainRunner

_refine_prompt_template = """
You are a helpful AI assistant, given the following conversation and a follow up request, rephrase the follow up request to be a standalone request, keeping the same user language.
Your rephrasing must include any relevant history element to get a precise standalone request and not losing previous context.

Chat History:
{chat_history}

Follow Up Input: {question}

Begin!

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
        self._chain = LLMChain(llm=self.llm, prompt=refine_prompt, verbose=self.verbose)

    def _run(self, event: PipelineEvent):
        chat_history = str(event.conversation)
        logger.debug(f"Question: {event.query}\nChat history: {chat_history}")
        resp = self._chain.run({"question": event.query, "chat_history": chat_history})
        return {"answer": resp}


def get_refine_chain(config, verbose=False, prompt_template=None):
    llm = get_llm(config)
    verbose = verbose or config.verbose
    return RefineQuery(llm=llm, verbose=verbose, prompt_template=prompt_template)
