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

import inspect
from collections.abc import Iterable
from typing import Union

from mlrun.artifacts import DocumentArtifact


class VectorStoreCollection:
    """
    A wrapper class for vector store collections with MLRun integration.

    This class wraps a vector store implementation (like Milvus, Chroma) and provides
    integration with MLRun context for document and artifact management. It delegates
    most operations to the underlying vector store while handling MLRun-specific
    functionality.

    The class implements attribute delegation through __getattr__ and __setattr__,
    allowing direct access to the underlying vector store's methods and attributes
    while maintaining MLRun integration.
    """

    def __init__(
        self,
        mlrun_context: Union["MlrunProject", "MLClientCtx"],  # noqa: F821
        collection_name: str,
        vector_store: "VectorStore",  # noqa: F821
    ):
        self._collection_impl = vector_store
        self._mlrun_context = mlrun_context
        self.collection_name = collection_name

    def __getattr__(self, name):
        # This method is called when an attribute is not found in the usual places
        # Forward the attribute access to _collection_impl
        return getattr(self._collection_impl, name)

    def __setattr__(self, name, value):
        if name in ["_collection_impl", "_mlrun_context"] or name in self.__dict__:
            # Use the base class method to avoid recursion
            super().__setattr__(name, value)
        else:
            # Forward the attribute setting to _collection_impl
            setattr(self._collection_impl, name, value)

    def add_documents(
        self,
        documents: list["Document"],  # noqa: F821
        **kwargs,
    ):
        """
        Add a list of documents to the collection.

        If the instance has an MLRun context, it will update the MLRun artifacts
        associated with the documents.

        Args:
            documents (list[Document]): A list of Document objects to be added.
            **kwargs: Additional keyword arguments to be passed to the underlying
                      collection implementation.

        Returns:
            The result of the underlying collection implementation's add_documents method.
        """
        if self._mlrun_context:
            for document in documents:
                mlrun_uri = document.metadata.get(
                    DocumentArtifact.METADATA_ARTIFACT_URI_KEY
                )
                if mlrun_uri:
                    artifact = self._mlrun_context.get_store_resource(mlrun_uri)
                    artifact.collection_add(self.collection_name)
                    self._mlrun_context.update_artifact(artifact)

        return self._collection_impl.add_documents(documents, **kwargs)

    def add_artifacts(self, artifacts: list[DocumentArtifact], splitter=None, **kwargs):
        """
        Add a list of DocumentArtifact objects to the vector store collection.

        Converts artifacts to LangChain documents, adds them to the vector store, and
        updates the MLRun context. If documents are split, the IDs are handled appropriately.

        Args:
            artifacts (list[DocumentArtifact]): List of DocumentArtifact objects to add
            splitter (optional): Document splitter to break artifacts into smaller chunks.
                If None, each artifact becomes a single document.
            **kwargs: Additional arguments passed to the underlying add_documents method.
                Special handling for 'ids' kwarg:
                - If provided and document is split, IDs are generated as "{original_id}_{i}"
                    where i starts from 1 (e.g., "doc1_1", "doc1_2", etc.)
                - If provided and document isn't split, original IDs are used as-is

        Returns:
            list: List of IDs for all added documents. When no custom IDs are provided:
                - Without splitting: Vector store generates IDs automatically
                - With splitting: Vector store generates separate IDs for each chunk
                When custom IDs are provided:
                - Without splitting: Uses provided IDs directly
                - With splitting: Generates sequential IDs as "{original_id}_{i}" for each chunk
        """
        all_ids = []
        user_ids = kwargs.pop("ids", None)

        if user_ids:
            if not isinstance(user_ids, Iterable):
                raise ValueError("IDs must be an iterable collection")
            if len(user_ids) != len(artifacts):
                raise ValueError(
                    "The number of IDs should match the number of artifacts"
                )
        for index, artifact in enumerate(artifacts):
            documents = artifact.to_langchain_documents(splitter)
            artifact.collection_add(self.collection_name)
            if self._mlrun_context:
                self._mlrun_context.update_artifact(artifact)
            if user_ids:
                num_of_documents = len(documents)
                if num_of_documents > 1:
                    ids_to_pass = [
                        f"{user_ids[index]}_{i}" for i in range(1, num_of_documents + 1)
                    ]
                else:
                    ids_to_pass = [user_ids[index]]
                kwargs["ids"] = ids_to_pass
            ids = self._collection_impl.add_documents(documents, **kwargs)
            all_ids.extend(ids)
        return all_ids

    def remove_from_artifact(self, artifact: DocumentArtifact):
        """
        Remove the current object from the given artifact's collection and update the artifact.

        Args:
            artifact (DocumentArtifact): The artifact from which the current object should be removed.
        """
        artifact.collection_remove(self.collection_name)
        if self._mlrun_context:
            self._mlrun_context.update_artifact(artifact)

    def delete_artifacts(self, artifacts: list[DocumentArtifact]):
        """
        Delete a list of DocumentArtifact objects from the collection.

        This method removes the specified artifacts from the collection and updates the MLRun context.
        The deletion process varies depending on the type of the underlying collection implementation.

        Args:
            artifacts (list[DocumentArtifact]): A list of DocumentArtifact objects to be deleted.

        Raises:
            NotImplementedError: If the delete operation is not supported for the collection implementation.
        """
        store_class = self._collection_impl.__class__.__name__.lower()
        for artifact in artifacts:
            artifact.collection_remove(self.collection_name)
            if self._mlrun_context:
                self._mlrun_context.update_artifact(artifact)

            if store_class == "milvus":
                expr = f"{DocumentArtifact.METADATA_SOURCE_KEY} == '{artifact.get_source()}'"
                return self._collection_impl.delete(expr=expr)
            elif store_class == "chroma":
                where = {DocumentArtifact.METADATA_SOURCE_KEY: artifact.get_source()}
                return self._collection_impl.delete(where=where)

            elif (
                hasattr(self._collection_impl, "delete")
                and "filter"
                in inspect.signature(self._collection_impl.delete).parameters
            ):
                filter = {
                    "metadata": {
                        DocumentArtifact.METADATA_SOURCE_KEY: artifact.get_source()
                    }
                }
                return self._collection_impl.delete(filter=filter)
            else:
                raise NotImplementedError(
                    f"delete_artifacts() operation not supported for {store_class}"
                )
