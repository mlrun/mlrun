import uuid
from pathlib import Path

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import AppConfig, get_vector_db, logger
from .web_loader import SmartWebLoader
from llmapps.controller.model import DocCollection
from llmapps.controller.sqlclient import client

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


# get the initialized loader class and its arguments from the type (web or file) and full file path
# use Path().suffix lib to extract the file extension from the file path
def get_loader_obj(doc_path: str, loader_type: str = None, **extra_args):
    if loader_type == "web":
        return WebBaseLoader([doc_path], **extra_args)
    elif loader_type == "eweb":
        return SmartWebLoader([doc_path], **extra_args)
    else:
        ext = Path(doc_path).suffix
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            return loader_class(doc_path, **{**loader_args, **extra_args})
        raise ValueError(f"Unsupported file extension '{ext}'")


class DataLoader:
    """Loads documents into a vector store.
    Example:

        data_loader = DataLoader(config)
        loader = get_loader_obj("https://milvus.io/docs/overview.md", loader_type="web")
        data_loader.load(loader, metadata={"xx": "web"})
    """

    def __init__(self, config: AppConfig, vector_store=None):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def load(self, loader, metadata: dict = None, version: int = None):
        """Loads documents into the vector store.

        Args:
            loader: A document loader.
            metadata: A dictionary of metadata to attach to the documents.
            version: A version number for the documents.
        """
        docs = loader.load()
        to_chunk = not hasattr(loader, "chunked")
        for doc in docs:
            self.ingest_document(doc, metadata, version, to_chunk=to_chunk)

    def ingest_document(
        self,
        doc,
        metadata: dict = None,
        version: int = None,
        doc_uid: str = None,
        to_chunk: bool = True,
    ):
        """Ingests a document into the vector store.

        Args:
            doc: A document.
            metadata: A dictionary of extra metadata to attach to the document.
            version: A version number for the document.
            doc_uid: A unique identifier for the document (will be generated if None).
        """
        if not doc_uid:
            doc_uid = uuid.uuid4().hex
        if to_chunk:
            chunks = self.text_splitter.split_documents([doc])
        else:
            chunks = [doc]
        for i, chunk in enumerate(chunks):
            if to_chunk:
                chunk.metadata["chunk"] = i
            if metadata:
                for key, value in metadata.items():
                    chunk.metadata[key] = value
            chunk.metadata["doc_uid"] = doc_uid
            if version:
                chunk.metadata["version"] = version
            logger.debug(
                f"Loading doc chunk:\n{chunk.page_content}\nMetadata: {chunk.metadata}"
            )
        self.vector_store.add_documents(chunks)


def get_data_loader(
    config: AppConfig, collection_name: str = None, session=None
) -> DataLoader:
    """Get a data loader instance."""
    close_session = True if session is None else False
    session = session or client.get_db_session()
    collection_name = collection_name or config.default_collection()
    db_args = None
    collection = client.get_collection(collection_name, session=session).data
    if not collection:
        client.create_collection(DocCollection(name=collection_name), session=session)
    else:
        db_args = collection.db_args
    if close_session:
        session.close()
    vector_db = get_vector_db(
        config,
        collection_name=collection_name,
        vector_store_args=db_args,
    )
    return DataLoader(config, vector_store=vector_db)
