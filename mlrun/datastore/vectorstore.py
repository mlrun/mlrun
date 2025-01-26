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
from typing import Optional, Union

from mlrun.artifacts import DocumentArtifact


def find_existing_attribute(obj, base_name="name", parent_name="collection"):
    # Define all possible patterns

    return None


def _extract_collection_name(vectorstore: "VectorStore") -> str:  # noqa: F821
    patterns = [
        "collection.name",
        "collection._name",
        "_collection.name",
        "_collection._name",
        "collection_name",
        "_collection_name",
    ]

    def resolve_attribute(obj, pattern):
        if "." in pattern:
            parts = pattern.split(".")
            current = vectorstore
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            return current
        else:
            return getattr(obj, pattern, None)

    if type(vectorstore).__name__ == "PineconeVectorStore":
        try:
            url = (
                vectorstore._index.config.host
                if hasattr(vectorstore._index, "config")
                else vectorstore._index._config.host
            )
            index_name = url.split("//")[1].split("-")[0]
            return index_name
        except Exception:
            pass

    for pattern in patterns:
        try:
            value = resolve_attribute(vectorstore, pattern)
            if value is not None:
                return value
        except (AttributeError, TypeError):
            continue

    # If we get here, we couldn't find a valid collection name
    raise ValueError(
        "Failed to extract collection name from the vector store. "
        "Please provide the collection name explicitly. "
    )


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
        vector_store: "VectorStore",  # noqa: F821
        collection_name: Optional[str] = None,
    ):
        self._collection_impl = vector_store
        self._mlrun_context = mlrun_context
        self.collection_name = collection_name or _extract_collection_name(vector_store)

    @property
    def __class__(self):
        # Make isinstance() check the wrapped object's class
        return self._collection_impl.__class__

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

    def _get_mlrun_project_name(self):
        import mlrun

        if self._mlrun_context and isinstance(
            self._mlrun_context, mlrun.projects.MlrunProject
        ):
            return self._mlrun_context.name
        if self._mlrun_context and isinstance(
            self._mlrun_context, mlrun.execution.MLClientCtx
        ):
            return self._mlrun_context.get_project_object().name
        return None

    def delete(self, *args, **kwargs):
        self._collection_impl.delete(*args, **kwargs)

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
                mlrun_key = document.metadata.get(
                    DocumentArtifact.METADATA_ARTIFACT_KEY, None
                )
                mlrun_project = document.metadata.get(
                    DocumentArtifact.METADATA_ARTIFACT_PROJECT, None
                )

                if mlrun_key and mlrun_project == self._get_mlrun_project_name():
                    mlrun_tag = document.metadata.get(
                        DocumentArtifact.METADATA_ARTIFACT_TAG, None
                    )
                    artifact = self._mlrun_context.get_artifact(
                        key=mlrun_key, tag=mlrun_tag
                    )
                    if artifact.collection_add(self.collection_name):
                        self._mlrun_context.update_artifact(artifact)

        return self._collection_impl.add_documents(documents, **kwargs)

    def add_artifacts(self, artifacts: list[DocumentArtifact], splitter=None, **kwargs):
        """
        Add a list of DocumentArtifact objects to the vector store collection.

        Converts artifacts to LangChain documents, adds them to the vector store, and
        updates the MLRun context. If documents are split, the IDs are handled appropriately.

        :param artifacts: List of DocumentArtifact objects to add
        :type artifacts: list[DocumentArtifact]
        :param splitter: Document splitter to break artifacts into smaller chunks.
                        If None, each artifact becomes a single document.
        :type splitter: TextSplitter, optional
        :param kwargs: Additional arguments passed to the underlying add_documents method.
                    Special handling for 'ids' kwarg:

                    * If provided and document is split, IDs are generated as "{original_id}_{i}"
                        where i starts from 1 (e.g., "doc1_1", "doc1_2", etc.)
                    * If provided and document isn't split, original IDs are used as-is

        :return: List of IDs for all added documents. When no custom IDs are provided:

                * Without splitting: Vector store generates IDs automatically
                * With splitting: Vector store generates separate IDs for each chunk

                When custom IDs are provided:

                * Without splitting: Uses provided IDs directly
                * With splitting: Generates sequential IDs as "{original_id}_{i}" for each chunk
        :rtype: list

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
            if artifact.collection_add(self.collection_name) and self._mlrun_context:
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

        if artifact.collection_remove(self.collection_name) and self._mlrun_context:
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
            if artifact.collection_remove(self.collection_name) and self._mlrun_context:
                self._mlrun_context.update_artifact(artifact)

            if store_class == "milvus":
                expr = f"{DocumentArtifact.METADATA_SOURCE_KEY} == '{artifact.get_source()}'"
                self._collection_impl.delete(expr=expr)
            elif store_class == "chroma":
                where = {DocumentArtifact.METADATA_SOURCE_KEY: artifact.get_source()}
                self._collection_impl.delete(where=where)
            elif store_class == "pineconevectorstore":
                filter = {
                    DocumentArtifact.METADATA_SOURCE_KEY: {"$eq": artifact.get_source()}
                }
                self._collection_impl.delete(filter=filter)
            elif store_class == "mongodbatlasvectorsearch":
                filter = {DocumentArtifact.METADATA_SOURCE_KEY: artifact.get_source()}
                self._collection_impl.collection.delete_many(filter=filter)
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
                self._collection_impl.delete(filter=filter)
            else:
                raise NotImplementedError(
                    f"delete_artifacts() operation not supported for {store_class}"
                )
