from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.schema.callbacks.base import BaseCallbackHandler

from ..config import get_llm, get_vector_db, logger
from ..schema import PipelineEvent
from .base import ChainRunner


class DocumentCallbackHandler(BaseCallbackHandler):
    """A callback handler that adds index number to documents retrieved."""

    def on_retriever_end(
        self,
        documents,
        *,
        run_id,
        parent_run_id,
        **kwargs,
    ):
        logger.debug(f"on_retriever: {documents}")
        if documents:
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
        self, llm, vector_store, verbose=False, chain_type: str = None, **search_kwargs
    ):
        document_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {index}",
            input_variables=["page_content", "index"],
        )

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            chain_type=chain_type or "stuff",  # "map_reduce",
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs=search_kwargs),
            return_source_documents=True,
            chain_type_kwargs={"document_prompt": document_prompt},
            verbose=verbose,
        )
        handler = DocumentCallbackHandler()
        handler.verbose = verbose
        self.chain_type = chain_type
        self.verbose = verbose
        self.cb = handler

    @classmethod
    def from_config(cls, config, collection_name: str = None, **search_kwargs):
        """Creates a document retriever from a config object."""
        vector_db = get_vector_db(config, collection_name=collection_name)
        llm = get_llm(config)
        return cls(llm, vector_db, verbose=config.verbose, **search_kwargs)

    def _get_answer(self, query):
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

    def run(self, event: PipelineEvent):
        # TODO: use text when is_cli
        logger.debug(f"Retriever Question: {event.query}\n")
        answer, sources = self._get_answer(event.query)
        logger.debug(f"answer: {answer} \nSources: {sources}")
        return {"answer": answer, "sources": sources}


class MultiRetriever(ChainRunner):

    def __init__(self, llm=None, default_collection=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.default_collection = default_collection
        self._retrievers = {}

    def post_init(self, mode="sync"):
        self.llm = self.llm or get_llm(self.context._config)
        if not self.default_collection:
            self.default_collection = self.context._config.default_collection()

    def _get_retriever(self, collection_name: str = None):
        collection_name = collection_name or self.default_collection
        logger.debug(f"Selected collection: {collection_name}")
        if collection_name not in self._retrievers:
            vector_db = get_vector_db(
                self.context._config, collection_name=collection_name
            )
            retriever = DocumentRetriever(
                self.llm,
                vector_db,
                verbose=self.verbose,
                # collection_name=collection_name,
            )
            self._retrievers[collection_name] = retriever

        return self._retrievers[collection_name]

    def _run(self, event: PipelineEvent):
        retriever = self._get_retriever(event.kwargs.get("collection_name"))
        return retriever.run(event)


def fix_milvus_filter_arg(vector_db, search_kwargs):
    """Fixes the milvus filter argument."""
    # detect if its Milvus and need to swap the filter dict arg with expr string
    if "filter" in search_kwargs and hasattr(vector_db, "_create_connection_alias"):
        filter = search_kwargs.pop("filter")
        if isinstance(filter, dict):
            # convert a dict of key value pairs to a string with key1=value1 and key2=value2
            filter = " and ".join(f"{k}={v}" for k, v in filter.items())
        search_kwargs["expr"] = filter


def get_retriever_from_config(
    config, verbose=False, collection_name: str = None, **search_kwargs
):
    """Creates a document retriever from a config object."""
    vector_db = get_vector_db(config, collection_name=collection_name)
    llm = get_llm(config)
    verbose = verbose or config.verbose
    return DocumentRetriever(llm, vector_db, verbose=verbose, **search_kwargs)
