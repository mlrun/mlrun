# Vector databases

Vector databases are used to enrich the context of a request before it is passed to a model for inference. This is a common practice in text processing tasks, where the context of a request can significantly impact the model's response. For example, in a conversational AI model, the context of the conversation can help the model understand the user's intent and provide a more accurate response. Another common scenario is using vector databases with RAG (Retrieval-Augmented Generation) models to retrieve relevant documents before generating a response.

Vector databases work by storing vectors that represent the context of a request. These vectors can be generated using various techniques, such as word embeddings. When a request is received, the vector database retrieves the vectors that represent the context of the request and passes them to the model for inference. This allows the model to take into account the context of the request and provide a more accurate response.

In MLRun, you can use vector databases to enrich the context of a request before passing it to a model for inference. This allows you to build more sophisticated models that take into account the context of the request and provide more accurate responses.

MLRun does not come with a VectorDB out-of-the-box: you need to install your choice of DB.

See also {ref}`genai-03-vectordb`.

## Using vector databases in MLRun

To use a vector database, you can create a function that stores the text data in the database. Then, typically, during the inference pipeline, you can retrieve the vectors from the database and enrich the context of the request before passing it to the model for inference.

For example, the following function adds data to a ChromaDB vector database:

```python
import mlrun
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mlrun.execution import MLClientCtx


def handler_chroma(
    context: MLClientCtx, vector_db_data: DataItem, cache_dir: str, collection_name: str
):

    df = vector_db_data.as_df()

    # Create chroma client
    chroma_client = chromadb.PersistentClient(path=cache_dir)

    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)

    # Add data to the collection
    collection = chroma_client.create_collection(name=collection_name)

    # Format and split documents
    documents = df.pop("page_content").to_list()
    metadatas = df.to_dict(orient="records")

    docs = [Document(page_content=d, metadata=m) for d, m in zip(documents, metadatas)]
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


project = mlrun.get_or_create_project("mlrun-with-chromadb-prj")

project.set_function(
    "ingest-to-chroma", kind="job", image="mlrun/mlrun", handler="handler_chroma"
)
```

Then, during inference, you might have a function that retrieves the documents of a specific topic. For example:

```python
collection = chroma_client.get_collection(collection_name)
results = collection.query(query_texts=[topic], n_results=10)
collection.query(query_texts=[topic], n_results=10)
q_context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
prompt_template = f"Relevant context: {q_context}\n\n The user's question: {question}"
```

## Supported vector databases

MLRun does not limit the choice of vector databases you can use. You can use any vector database that fits your use case. Some popular vector databases include:
- [ChromaDB](https://github.com/chroma-core/chroma)
- [milvus](https://github.com/milvus-io/milvus)
- [MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search)
- [Pinecone](https://www.pinecone.io/)

These databases provide different features and capabilities, so you can choose the one that best fits your use case.