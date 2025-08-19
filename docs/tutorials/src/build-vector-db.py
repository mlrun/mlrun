import re

import chromadb
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import mlrun
from mlrun.execution import MLClientCtx


def ensure_alphanumeric_end(s):
    return s if re.search(r"[a-zA-Z0-9]$", s) else s + "a"


@mlrun.handler()
def handler_chroma(
    context: MLClientCtx,
    df: pd.DataFrame,
    cache_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    collection_name: str = "my_news",
):
    # project = mlrun.get_current_project()

    spec = mlrun.artifacts.DocumentLoaderSpec(
        loader_class_name="langchain_community.document_loaders.WebBaseLoader",
        src_name="web_path",
        download_object=False,
    )

    if cache_dir.startswith("s3://"):
        cache_dir = "./"

    # Create chroma client
    chroma_client = chromadb.PersistentClient(path=cache_dir)

    # Get or create collection
    collection_name = collection_name
    print(f"Creating collection: '{collection_name}'")

    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Format and split docunments
    documents = df.pop("page_content").to_list()
    metadatas = df.to_dict(orient="records")
    docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(documents, metadatas)
        if type(d) is str
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    for doc in splits:
        # Make sure artifact key ends with alpha-numeric char
        artifact_key = ensure_alphanumeric_end(
            mlrun.artifacts.DocumentArtifact.key_from_source(doc.metadata["link"])
        )

        collection.add(
            ids=[artifact_key],
            metadatas=[doc.metadata],
            documents=[doc.page_content],
        )

        context.log_document(
            key=artifact_key,
            target_path=doc.metadata["link"],
            document_loader_spec=spec,
        )

    vectordb = context.log_model(
        "vect_db",
        artifact_path=context.artifact_subpath("vect_db"),
        model_file=f"{cache_dir}/chroma.sqlite3",
    )

    context.logger.info(f"Vector DB was created {vectordb}")
