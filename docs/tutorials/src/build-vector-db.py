import uuid

import chromadb
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import mlrun
from mlrun.execution import MLClientCtx


@mlrun.handler()
def handler_chroma(
    context: MLClientCtx,
    df: pd.DataFrame,
    cache_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
):
    # Create chroma client
    chroma_client = chromadb.PersistentClient(path=cache_dir)

    # Get or create collection
    collection_name = "my_news"
    print(f"Creating collection: '{collection_name}'")

    if len(chroma_client.list_collections()) > 0 and collection_name in [
        chroma_client.list_collections()[0].name
    ]:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.create_collection(name=collection_name)

    # Format and split docunments
    documents = df.pop("page_content").to_list()
    metadatas = df.to_dict(orient="records")
    docs = [Document(page_content=d, metadata=m) for d, m in zip(documents, metadatas)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    # Add to vector store
    collection.add(
        ids=[str(uuid.uuid4()) for d in splits],
        metadatas=[d.metadata for d in splits],
        documents=[d.page_content for d in splits],
    )

    context.logger.info("Vector DB was created")
