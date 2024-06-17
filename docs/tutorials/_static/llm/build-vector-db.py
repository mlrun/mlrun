import pandas as pd
import mlrun
from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import numpy as np
import chromadb

def handler_chroma(context:MLClientCtx, vector_db_data: DataItem, cache_dir: str):
    
    df = vector_db_data.as_df().head(1000)

    # Create chroma client
    chroma_client = chromadb.PersistentClient(path=cache_dir)

    # Add data to the collection
    collection_name = "my_news"
    print(f"Creating collection: '{collection_name}'")
    
    if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)
    
    collection = chroma_client.create_collection(name=collection_name)

    collection.add(
        documents=df["title"][:100].tolist(),
        metadatas=[{"topic": topic} for topic in df["topic"][:100].tolist()],
        ids=[f"id{x}" for x in range(100)])
    
    context.logger.info("Vector DB was created")
