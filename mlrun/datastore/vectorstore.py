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
from importlib import import_module
from typing import Union

from mlrun.artifacts import DocumentArtifact


class VectorStoreCollection:
    """
    VectorStoreCollection is a class that manages a collection of vector stores, providing methods to add and delete
    documents and artifacts, and to interact with an MLRun context.

    Attributes:
        _collection_impl (object): The underlying collection implementation.
        _mlrun_context (Union[MlrunProject, MLClientCtx]): The MLRun context associated with the collection.
        collection_name (str): The name of the collection.
        id (str): The unique identifier of the collection, composed of the datastore profile and collection name.

    Methods:
        add_documents(documents: list["Document"], **kwargs):
            Adds a list of documents to the collection and updates the MLRun artifacts associated with the documents
            if an MLRun context is present.

        add_artifacts(artifacts: list[DocumentArtifact], splitter=None, **kwargs):
            Adds a list of DocumentArtifact objects to the collection, optionally using a splitter to convert
            artifacts to documents.

        remove_itself_from_artifact(artifact: DocumentArtifact):
            Removes the current object from the given artifact's collection and updates the artifact.

        delete_artifacts(artifacts: list[DocumentArtifact]):
            Deletes a list of DocumentArtifact objects from the collection and updates the MLRun context.
            Raises NotImplementedError if the delete operation is not supported for the collection implementation.
    """

    def __init__(
        self,
        vector_store_class: str,
        mlrun_context: Union["MlrunProject", "MLClientCtx"],  # noqa: F821
        datastore_profile: str,
        collection_name: str,
        **kwargs,
    ):
        # Import the vector store class dynamically
        module_name, class_name = vector_store_class.rsplit(".", 1)
        module = import_module(module_name)
        vector_store_class = getattr(module, class_name)

        signature = inspect.signature(vector_store_class)

        # Create the vector store instance
        if "collection_name" in signature.parameters.keys():
            vector_store = vector_store_class(collection_name=collection_name, **kwargs)
        else:
            vector_store = vector_store_class(**kwargs)

        self._collection_impl = vector_store
        self._mlrun_context = mlrun_context
        self.collection_name = collection_name
        self.id = datastore_profile + "/" + collection_name

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
                    artifact.collection_add(self.id)
                    self._mlrun_context.update_artifact(artifact)
        return self._collection_impl.add_documents(documents, **kwargs)

    def add_artifacts(self, artifacts: list[DocumentArtifact], splitter=None, **kwargs):
        """
        Add a list of DocumentArtifact objects to the collection.

        Args:
            artifacts (list[DocumentArtifact]): A list of DocumentArtifact objects to be added.
            splitter (optional): An optional splitter to be used when converting artifacts to documents.
            **kwargs: Additional keyword arguments to be passed to the collection's add_documents method.

        Returns:
            list: A list of IDs of the added documents.
        """
        all_ids = []
        for artifact in artifacts:
            documents = artifact.to_langchain_documents(splitter)
            artifact.collection_add(self.id)
            self._mlrun_context.update_artifact(artifact)
            ids = self._collection_impl.add_documents(documents, **kwargs)
            all_ids.extend(ids)
        return all_ids

    def remove_itself_from_artifact(self, artifact: DocumentArtifact):
        """
        Remove the current object from the given artifact's collection and update the artifact.

        Args:
            artifact (DocumentArtifact): The artifact from which the current object should be removed.
        """
        artifact.collection_remove(self.id)
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
            artifact.collection_remove(self.id)
            self._mlrun_context.update_artifact(artifact)
            if store_class == "milvus":
                expr = f"{DocumentArtifact.METADATA_SOURCE_KEY} == '{artifact.source}'"
                return self._collection_impl.delete(expr=expr)
            elif store_class == "chroma":
                where = {DocumentArtifact.METADATA_SOURCE_KEY: artifact.source}
                return self._collection_impl.delete(where=where)

            elif (
                hasattr(self._collection_impl, "delete")
                and "filter"
                in inspect.signature(self._collection_impl.delete).parameters
            ):
                filter = {
                    "metadata": {DocumentArtifact.METADATA_SOURCE_KEY: artifact.source}
                }
                return self._collection_impl.delete(filter=filter)
            else:
                raise NotImplementedError(
                    f"delete_artifacts() operation not supported for {store_class}"
                )
