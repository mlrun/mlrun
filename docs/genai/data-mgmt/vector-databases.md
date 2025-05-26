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

## Experiment tracking with a vector DB
MLRun enables experiment tracking for document-based models, using the LangChain API to integrate directly with vector databases. You can track documents as artifacts, complete with metadata such as loader type, producer information, and collection details.

The following example uses [Milvus](https://milvus.io/)

### Milvus configuration
Create and register a profile representing a Milvus DB. This is done in the project level, once per project. Credentials for the DB may be passed here assuming the code is not introduced into any repo, or they may be provided through project secrets. See [ConfigProfile](https://docs.mlrun.org/en/stable/api/mlrun.datastore/index.html#mlrun.datastore.datastore_profile.ConfigProfile)
```python
import mlrun
import tempfile
from langchain.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from mlrun.artifacts import DocumentLoaderSpec, MLRunLoader
from mlrun.datastore.datastore_profile import (
    ConfigProfile,
    register_temporary_client_datastore_profile,
)

profile = ConfigProfile(
    name="milvus-config", public={"MILVUS_DB": {"host": "localhost", "port": 19530}}
)
# Register the profile temporarily for the current client session
register_temporary_client_datastore_profile(profile)
```

### Creating an MLRun collection from Milvus
Create (or use an existing) collection to store the artifact/documents in. Use the configuration stored earlier in the ConfigProfile to get the configuration details. You still need to create the actual VectorDB class, since each VectorDB has a different initialization method. See [get_config_profile_attribute](https://docs.mlrun.org/en/stable/api/mlrun.projects/index.html#mlrun.projects.MlrunProject.get_config_profile_attributes)
```python
# Initialize embedding model (using FakeEmbeddings for demonstration)
embedding_model = FakeEmbeddings(size=3)

config = project.get_config_profile_attributes("milvus-config")
```

### Creating the Milvus vector store
In this step you also create the MLRun collection wrapper. See [get_vector_store_collection](https://docs.mlrun.org/en/stable/api/mlrun.projects/index.html#mlrun.projects.MlrunProject.get_vector_store_collection).
```python
vectorstore = Milvus(
    collection_name="my_tutorial_collection",
    embedding_function=embedding_model,
    connection_args=config["MILVUS_DB"],
    auto_id=True,
)

# Create MLRun collection wrapper
collection = project.get_vector_store_collection(vector_store=vectorstore)
```

### MLRun document artifact
You can add documents to the collection using either [add_documents()](https://docs.mlrun.org/en/latest/api/mlrun.datastore/index.html#mlrun.datastore.vectorstore.VectorStoreCollection.add_documents), which accepts LangChain document objects, or [add_artifacts()](https://docs.mlrun.org/en/latest/api/mlrun.datastore/index.html#mlrun.datastore.vectorstore.VectorStoreCollection.add_artifacts), which generates documents from artifacts and inserts them into the collection.
```python
# Create a sample document
def create_sample_document(content, dir=None):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=dir
    ) as temp_file:
        temp_file.write(content)
        return temp_file.name


# Create and log an MLRun artifact
file_path = create_sample_document("Sample content for demonstration")
artifact = project.log_document("sample-doc", local_path=file_path)

# Convert MLRun artifact to LangChain documents
langchain_docs = artifact.to_langchain_documents()
print("LangChain document content:", langchain_docs[0].page_content)
print("LangChain document metadata:", langchain_docs[0].metadata)

# Add LangChain documents to collection
milvus_ids = collection.add_documents(langchain_docs)
print("Documents added with IDs:", milvus_ids)

# Search in collection
results = collection.similarity_search("sample", k=1)
print("Search results:", [doc.page_content for doc in results])
```

> LangChain document content: Sample content for demonstration
> LangChain document metadata: {'source': 'vectorstore-demo3/sample-doc', 'original_source': '/tmp/tmpawvdmdq5.txt', 'mlrun_object_uri': 'store://artifacts/vectorstore-demo3/sample-doc#0@eb00adb2de8042c4eae0d7b18b4a3797c2749dac^b9ab6e6eca457947c6a0cbfeeb456a71cd2e8798', 'mlrun_chunk': '0'}  
> Documents added with IDs: [454354693163582836]  
> Search results: ['Sample content for demonstration']


```python
# Add artifacts directly to collection
artifact1 = project.log_document(
    "doc1", local_path=create_sample_document("First document")
)
artifact2 = project.log_document(
    "doc2", local_path=create_sample_document("Second document")
)

# Add multiple artifacts at once
milvus_ids = collection.add_artifacts([artifact1, artifact2])
print("Artifacts added with IDs:", milvus_ids)

# Get back as LangChain documents
search_results = collection.similarity_search("first")
print("Retrieved document:", search_results[0].page_content)
```
> Artifacts added with IDs: [454354693163582838, 454354693163582840]
> Retrieved document: First document


### Using MLRunLoader
MLRunLoader is a wrapper. It receives langchain loader as a parameter (for example `Langchain.PDFloader`, or `Langchain.CSVLoader`), and calls this underlying loader for all its purposes. In addition, it adds the source file as an MLRun artifact.
```python
# Create a document loader specification
loader_spec = DocumentLoaderSpec(
    loader_class_name="langchain_community.document_loaders.TextLoader",
    src_name="file_path",
)

# Create and use MLRunLoader
file_path = create_sample_document("Content for MLRunLoader test")
loader = MLRunLoader(
    source_path=file_path,
    loader_spec=loader_spec,
    artifact_key="loaded-doc",
    producer=project,
)

# Load documents
documents = loader.load()
print("Loaded document content:", documents[0].page_content)

# Verify artifact creation
artifact = project.get_artifact("loaded-doc")
print("Created artifact key:", artifact.key)
```

> Loaded document content: Content for MLRunLoader test  
> Created artifact key: loaded-doc
 
### Using MLRunLoader with DirectoryLoader
Langchain DirectoryLoader loads all the files in the directory by calling the langchain loader. When you pass MLRunLoader, all the source files are added as MLRun artifacts. See [DocumentLoaderSpec](https://docs.mlrun.org/en/stable/api/mlrun.artifacts/mlrun.artifacts.document.html#mlrun.artifacts.document.DocumentLoaderSpec).



```python
# Create a directory with multiple documents
temp_dir = tempfile.mkdtemp()
create_sample_document("First file content", dir=temp_dir)
create_sample_document("Second file content", dir=temp_dir)

# Configure loader specification
artifact_loader_spec = DocumentLoaderSpec(
    loader_class_name="langchain_community.document_loaders.TextLoader",
    src_name="file_path",
)

# Create directory loader with MLRunLoader
dir_loader = DirectoryLoader(
    temp_dir,
    glob="**/*.*",
    loader_cls=MLRunLoader,
    loader_kwargs={
        "loader_spec": artifact_loader_spec,
        "artifact_key": "dir_doc%%",  # %% will be replaced with unique identifier
        "producer": project,
        "upload": False,
    },
)

# Load all documents
documents = dir_loader.load()
print(f"Loaded {len(documents)} documents")

# List created artifacts
artifacts = project.list_artifacts(kind="document")
matching_artifacts = [
    art for art in artifacts if art["metadata"]["key"].startswith("dir_doc")
]

print("Created artifacts:", [art["metadata"]["key"] for art in matching_artifacts])
```
> Loaded 2 documents  
> Created artifacts: ['dir_doc2ftmp2ftmppsy90gh72ftmp6y86mt4d.txt', 'dir_doc2ftmp2ftmppsy90gh72ftmp2w62cehp.txt']


You can check the full tutorial notebook [here](https://github.com/mlrun/mlrun/blob/development/docs/tutorials/genai-03-vector-db.ipynb)

## Supported vector databases

MLRun does not limit the choice of vector databases you can use. You can use any vector database that fits your use case. Some popular vector databases include:
- [ChromaDB](https://github.com/chroma-core/chroma)
- [milvus](https://github.com/milvus-io/milvus)
- [MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search)
- [Pinecone](https://www.pinecone.io/)

These databases provide different features and capabilities, so you can choose the one that best fits your use case.