# Copyright 2024 Iguazio
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
#

import os
import tempfile

import pytest
import yaml

from mlrun.artifacts import DocumentLoader, DocumentLoaderSpec
from mlrun.datastore.datastore_profile import (
    VectorStoreProfile,
)
from tests.system.base import TestMLRunSystem

here = os.path.dirname(__file__)
config_file_path = os.path.join(here, "../env.yml")

config = {}
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file)


@pytest.mark.skipif(
    not config.get("MILVUS_HOST") or not config.get("MILVUS_PORT"),
    reason="milvus parameters not configured",
)
# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestDatastoreProfile(TestMLRunSystem):
    def custom_setup(self):
        pass

    def test_vectorstore_document_artifact(self):
        # Create a temporary text file with a simple context
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("This is a test document for vector store.")
            temp_file.flush()
            # Test logging a document localy
            artifact = self.project.log_document(
                "test_document_artifact", src_path=temp_file.name, upload=False
            )
            langchain_documents = artifact.to_langchain_documents()

            assert len(langchain_documents) == 1
            assert (
                langchain_documents[0].page_content
                == "This is a test document for vector store."
            )
            assert (
                langchain_documents[0].metadata["source"]
                == f"{self.project.name}/test_document_artifact"
            )
            assert langchain_documents[0].metadata["original_source"] == temp_file.name
            assert langchain_documents[0].metadata["mlrun_object_uri"] == artifact.uri
            assert langchain_documents[0].metadata["mlrun_chunk"] == "0"

            # Test logging a document localy
            artifact = self.project.log_document(
                "test_document_artifact", src_path=temp_file.name, upload=True
            )

            stored_artifcat = self.project.get_artifact("test_document_artifact")
            stored_langchain_documents = stored_artifcat.to_langchain_documents()
            stored_langchain_documents[0].metadata["mlrun_target_path"]

            assert (
                langchain_documents[0].page_content
                == stored_langchain_documents[0].page_content
            )
            assert (
                langchain_documents[0].metadata["source"]
                == stored_langchain_documents[0].metadata["source"]
            )
            assert (
                langchain_documents[0].metadata["original_source"]
                == stored_langchain_documents[0].metadata["original_source"]
            )
            assert (
                langchain_documents[0].metadata["mlrun_chunk"]
                == stored_langchain_documents[0].metadata["mlrun_chunk"]
            )
            assert (
                stored_langchain_documents[0].metadata["mlrun_object_uri"]
                == stored_artifcat.uri
            )
            assert (
                stored_langchain_documents[0].metadata["mlrun_target_path"]
                == stored_artifcat.get_target_path()
            )

    def test_vectorstore_document_mlrun_artifact(self):
        # Check mlrun loader
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("This is a test document for vector store.")
            temp_file.flush()
            # Test logging a document localy
            loader = DocumentLoader(
                source_path=temp_file.name,
                loader_spec=DocumentLoaderSpec(),
                artifact_key="test_document_artifact",
                producer=self.project,
                upload=False,
            )
            lc_documents = loader.load()
            assert len(lc_documents) == 1

    def test_vectorstore_collection_documents(self):
        from langchain.embeddings import FakeEmbeddings

        embedding_model = FakeEmbeddings(size=3)
        profile = VectorStoreProfile(
            name="milvus",
            vector_store_class="langchain.vectorstores.Milvus",
            kwargs_private={
                "connection_args": {
                    "host": config["MILVUS_HOST"],
                    "port": config["MILVUS_PORT"],
                }
            },
        )
        collection = self.project.get_or_create_vector_store_collection(
            collection_name="collection_name",
            profile=profile,
            embedding_function=embedding_model,
            auto_id=True,
        )
        with tempfile.NamedTemporaryFile(mode="w") as temp_file1:
            temp_file1.write(
                "Machine learning enables computers to learn from data without being explicitly programmed."
            )
            temp_file1.flush()
            with tempfile.NamedTemporaryFile(mode="w") as temp_file2:
                temp_file2.write(
                    "Machine learning enables computers to learn from data without being explicitly programmed."
                )
                temp_file2.flush()
                with tempfile.NamedTemporaryFile(mode="w") as temp_file3:
                    temp_file3.write(
                        "Machine learning enables computers to learn from data without being explicitly programmed."
                    )
                    temp_file3.flush()

                    doc1 = self.project.log_document(
                        "lc_doc1", src_path=temp_file1.name, upload=False
                    )
                    doc2 = self.project.log_document(
                        "lc_doc2", src_path=temp_file2.name, upload=False
                    )
                    doc3 = self.project.log_document(
                        "lc_doc3", src_path=temp_file3.name, upload=False
                    )

                    milvus_ids = collection.add_artifacts([doc1, doc2])
                    assert len(milvus_ids) == 2

                    direct_milvus_id = collection.add_documents(
                        doc3.to_langchain_documents()
                    )
                    assert len(direct_milvus_id) == 1
                    milvus_ids.append(direct_milvus_id[0])

                    collection.col.flush()
                    documents_in_db = collection.similarity_search(
                        query="",
                        expr=f"{doc1.METADATA_ORIGINAL_SOURCE_KEY} == '{temp_file1.name}'",
                    )
                    assert len(documents_in_db) == 1

                    collection.delete_artifacts([doc1])
                    collection.col.flush()

                    documents_in_db = collection.similarity_search(
                        query="",
                        expr=f"{doc1.METADATA_ORIGINAL_SOURCE_KEY} == '{temp_file1.name}'",
                    )
                    assert len(documents_in_db) == 0

        collection.col.drop()
