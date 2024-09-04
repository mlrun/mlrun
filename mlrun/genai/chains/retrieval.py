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

import logging
from typing import Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from mlrun.genai.chains.base import ChainRunner
from mlrun.genai.config import get_llm, get_vector_db
from mlrun.genai.schemas import WorkflowEvent

logger = logging.getLogger(__name__)


class DocumentCallbackHandler(BaseCallbackHandler):
    """Callback handler that adds index numbers to retrieved documents."""

    def on_retriever_end(self, documents: List[Document], **kwargs):
        logger.debug(f"Retrieved documents: {documents}")
        for i, doc in enumerate(documents):
            doc.metadata["index"] = str(i)


class DocumentRetriever:
    """A wrapper for the retrieval QA chain that returns source documents.

    Example:
        vector_store = get_vector_db(config)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        query = "What is an llm?"
        dr = document_retrevial(llm, vector_store)
        dr.get_answer(query)

    Args:
        llm: A language model.
        vector_store: A vector store.
        verbose: Whether to print debug information.

    """

    def __init__(
        self,
        llm,
        vector_store,
        verbose: bool = False,
        chain_type: Optional[str] = None,
        **search_kwargs,
    ):
        document_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {index}",
            input_variables=["page_content", "index"],
        )

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs=search_kwargs),
            chain_type=chain_type or "stuff",  # "map_reduce",
            return_source_documents=True,
            chain_type_kwargs={"document_prompt": document_prompt},
            verbose=verbose,
        )
        self.cb = DocumentCallbackHandler()
        self.cb.verbose = verbose
        self.verbose = verbose
        self.chain_type = chain_type  # TODO check what this is

    @classmethod
    def from_config(
        cls, config, collection_name: Optional[str] = None, **search_kwargs
    ):
        """Create a document retriever from a config object."""
        vector_db = get_vector_db(config, collection_name=collection_name)
        llm = get_llm(config)
        return cls(llm, vector_db, verbose=config.verbose, **search_kwargs)

    def _get_answer(self, query: str) -> tuple[str, List[Document]]:
        """Get the answer to a question and the source documents used."""
        result = self.chain({"question": query}, callbacks=[self.cb])
        sources = [s.strip() for s in result["sources"].split(",")]
        source_docs = [
            doc
            for doc in result["source_documents"]
            if doc.metadata.pop("index", "") in sources
        ]
        if self.verbose:
            docs_string = "\n".join(str(doc.metadata) for doc in source_docs)
            logger.info(f"Source documents:\n{docs_string}")
        return result["answer"], source_docs

    def run(self, event: WorkflowEvent) -> Dict[str, any]:
        """Run the retrieval with the given event."""
        # TODO: use text when is_cli
        logger.debug(f"Retriever Question: {event.query}")
        # event.query.content is not always present
        query = event.query.content if hasattr(event.query, "content") else event.query
        answer, sources = self._get_answer(query)
        logger.debug(f"Answer: {answer}\nSources: {sources}")
        return {"answer": answer, "sources": sources}


class MultiRetriever(ChainRunner):
    """A class that manages multiple document retrievers."""

    def __init__(self, llm=None, default_collection: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.default_collection = default_collection
        self._retrievers: Dict[str, DocumentRetriever] = {}

    def post_init(self, mode: str = "sync"):
        self.llm = self.llm or get_llm(self.context._config)
        if not self.default_collection:
            self.default_collection = self.context._config.default_collection()

    def _get_retriever(
        self, collection_name: Optional[str] = None
    ) -> DocumentRetriever:
        collection_name = collection_name or self.default_collection
        logger.debug(f"Selected collection: {collection_name}")
        if collection_name not in self._retrievers:
            vector_db = get_vector_db(
                self.context._config, collection_name=collection_name
            )
            retriever = DocumentRetriever(self.llm, vector_db, verbose=self.verbose)
            self._retrievers[collection_name] = retriever
        return self._retrievers[collection_name]

    def _run(self, event: WorkflowEvent) -> Dict[str, any]:
        """Run the multi retriever."""
        collection_name = event.kwargs.get(
            "collection_name"
        )  # TODO name always in kwargs?
        retriever = self._get_retriever(collection_name)
        return retriever.run(event)


def fix_milvus_filter_arg(vector_db, search_kwargs: Dict[str, any]):
    """Fix the Milvus filter argument if necessary."""
    if "filter" in search_kwargs and hasattr(vector_db, "_create_connection_alias"):
        filter_arg = search_kwargs.pop("filter")
        if isinstance(filter_arg, dict):
            filter_str = " and ".join(f"{k}={v}" for k, v in filter_arg.items())
        else:
            filter_str = filter_arg
        search_kwargs["expr"] = filter_str


def get_retriever_from_config(
    config,
    verbose: bool = False,
    collection_name: Optional[str] = None,
    **search_kwargs,
) -> DocumentRetriever:
    """Create a document retriever from a config object."""
    vector_db = get_vector_db(config, collection_name=collection_name)
    llm = get_llm(config)
    verbose = verbose or config.verbose
    return DocumentRetriever(llm, vector_db, verbose=verbose, **search_kwargs)
