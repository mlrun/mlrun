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

import ast
import re
import tempfile
from collections.abc import Iterator
from copy import deepcopy
from importlib import import_module
from typing import Optional, Union

import mlrun
from mlrun.artifacts import Artifact, ArtifactSpec
from mlrun.model import ModelObj

from ..utils import generate_artifact_uri


class DocumentLoaderSpec(ModelObj):
    """
    A class to load a document from a file path using a specified loader class.

    This class is responsible for loading documents from a given source path using a specified loader class.
    The loader class is dynamically imported and instantiated with the provided arguments. The loaded documents
    can be optionally uploaded as artifacts.

    Attributes:
        loader_class_name (str): The name of the loader class to use for loading documents.
        src_name (str): The name of the source attribute to pass to the loader class.
        kwargs (Optional[dict]): Additional keyword arguments to pass to the loader class.

    Methods:
        make_loader(src_path): Creates an instance of the loader class with the specified source path.
    """

    _dict_fields = ["loader_class_name", "src_name", "kwargs"]

    def __init__(
        self,
        loader_class_name: str = "langchain_community.document_loaders.TextLoader",
        src_name: str = "file_path",
        kwargs: Optional[dict] = None,
    ):
        """
        Initialize the document loader.

        Args:
            loader_class_name (str): The name of the loader class to use.
            src_name (str): The source name for the document.
            kwargs (Optional[dict]): Additional keyword arguments to pass to the loader class.
        """
        self.loader_class_name = loader_class_name
        self.src_name = src_name
        self.kwargs = kwargs

    def make_loader(self, src_path):
        module_name, class_name = self.loader_class_name.rsplit(".", 1)
        module = import_module(module_name)
        loader_class = getattr(module, class_name)
        kwargs = deepcopy(self.kwargs or {})
        kwargs[self.src_name] = src_path
        loader = loader_class(**kwargs)
        return loader


class DocumentLoader:
    """
    A factory class for creating instances of a dynamically defined document loader.

    Args:
        artifact_key (str): The key for the artifact to be logged.It can include '%%' which will be replaced
        by a hex-encoded version of the source path.
        source_path (str): The source path of the document to be loaded.
        loader_spec (DocumentLoaderSpec): Specification for the document loader.
        producer (Optional[Union[MlrunProject, str, MLClientCtx]], optional): The producer of the document
        upload (bool, optional): Flag indicating whether to upload the document.

    Returns:
        DynamicDocumentLoader: An instance of a dynamically defined subclass of BaseLoader.
    """

    def __new__(
        cls,
        source_path: str,
        loader_spec: "DocumentLoaderSpec",
        artifact_key="doc%%",
        producer: Optional[Union["MlrunProject", str, "MLClientCtx"]] = None,  # noqa: F821
        upload: bool = False,
    ):
        # Dynamically import BaseLoader
        from langchain_community.document_loaders.base import BaseLoader

        class DynamicDocumentLoader(BaseLoader):
            def __init__(
                self,
                source_path,
                loader_spec,
                artifact_key,
                producer,
                upload,
            ):
                self.producer = producer
                self.artifact_key = (
                    DocumentLoader.artifact_key_instance(artifact_key, source_path)
                    if "%%" in artifact_key
                    else artifact_key
                )
                self.loader_spec = loader_spec
                self.source_path = source_path
                self.upload = upload

                # Resolve the producer
                if not self.producer:
                    self.producer = mlrun.mlconf.default_project
                if isinstance(self.producer, str):
                    self.producer = mlrun.get_or_create_project(self.producer)

            def lazy_load(self) -> Iterator["Document"]:  # noqa: F821
                artifact = self.producer.log_document(
                    key=self.artifact_key,
                    document_loader=self.loader_spec,
                    src_path=self.source_path,
                    upload=self.upload,
                )
                yield artifact.to_langchain_documents()

        # Return an instance of the dynamically defined subclass
        instance = DynamicDocumentLoader(
            artifact_key=artifact_key,
            source_path=source_path,
            loader_spec=loader_spec,
            producer=producer,
            upload=upload,
        )
        return instance

    @staticmethod
    def artifact_key_instance(artifact_key: str, src_path: str) -> str:
        if "%%" in artifact_key:
            pattern = mlrun.utils.regex.artifact_key[0]
            # Convert anchored pattern (^...$) to non-anchored version for finditer
            search_pattern = pattern.strip("^$")
            result = []
            current_pos = 0

            # Find all valid sequences
            for match in re.finditer(search_pattern, src_path):
                # Add hex values for characters between matches
                for char in src_path[current_pos : match.start()]:
                    result.append(hex(ord(char))[2:].zfill(2))

                # Add the valid sequence
                result.append(match.group())
                current_pos = match.end()

            # Handle any remaining characters after the last match
            for char in src_path[current_pos:]:
                result.append(hex(ord(char))[2:].zfill(2))

            resolved_path = "".join(result)

            artifact_key = artifact_key.replace("%%", resolved_path)

        return artifact_key


class DocumentArtifact(Artifact):
    """
    A specific artifact class inheriting from generic artifact, used to maintain Document meta-data.

    Methods:
        to_langchain_documents(splitter): Create LC documents from the artifact.
        collection_add(collection_id): Add a collection ID to the artifact.
        collection_remove(collection_id): Remove a collection ID from the artifact.
    """

    class DocumentArtifactSpec(ArtifactSpec):
        _dict_fields = ArtifactSpec._dict_fields + [
            "document_loader",
            "collections",
            "original_source",
        ]

        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.document_loader = None
            self.collections = set()
            self.original_source = None

    """
    A specific artifact class inheriting from generic artifact, used to maintain Document meta-data.
    """

    kind = "document"

    METADATA_SOURCE_KEY = "source"
    METADATA_ORIGINAL_SOURCE_KEY = "original_source"
    METADATA_CHUNK_KEY = "mlrun_chunk"
    METADATA_ARTIFACT_URI_KEY = "mlrun_object_uri"
    METADATA_ARTIFACT_TARGET_PATH_KEY = "mlrun_target_path"

    def __init__(
        self,
        key=None,
        document_loader: DocumentLoaderSpec = DocumentLoaderSpec(),
        **kwargs,
    ):
        super().__init__(key, **kwargs)
        self.spec.document_loader = document_loader.to_str()
        if "src_path" in kwargs:
            self.spec.original_source = kwargs["src_path"]

    @property
    def spec(self) -> DocumentArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(
            spec, "spec", DocumentArtifact.DocumentArtifactSpec
        )
        # _verify_dict doesn't handle set, so we need to convert it back
        if isinstance(self._spec.collections, str):
            self._spec.collections = ast.literal_eval(self._spec.collections)

    @property
    def inputs(self):
        # To keep the interface consistent with the project.update_artifact() when we update the artifact
        return None

    @property
    def source(self):
        return generate_artifact_uri(self.metadata.project, self.spec.db_key)

    def to_langchain_documents(
        self,
        splitter: Optional["TextSplitter"] = None,  # noqa: F821
    ) -> list["Document"]:  # noqa: F821
        from langchain.schema import Document

        """
        Create LC documents from the artifact

        Args:
            splitter (Optional[TextSplitter]): A LangChain TextSplitter to split the document into chunks.

        Returns:
            list[Document]: A list of LangChain Document objects.
        """
        dictionary = ast.literal_eval(self.spec.document_loader)
        loader_spec = DocumentLoaderSpec.from_dict(dictionary)

        if self.get_target_path():
            with tempfile.NamedTemporaryFile() as tmp_file:
                mlrun.datastore.store_manager.object(
                    url=self.get_target_path()
                ).download(tmp_file.name)
                loader = loader_spec.make_loader(tmp_file.name)
                documents = loader.load()
        elif self.src_path:
            loader = loader_spec.make_loader(self.src_path)
            documents = loader.load()
        else:
            raise ValueError(
                "No src_path or target_path provided. Cannot load document."
            )

        results = []
        for document in documents:
            if splitter:
                texts = splitter.split_text(document.page_content)
            else:
                texts = [document.page_content]

            metadata = document.metadata

            metadata[self.METADATA_ORIGINAL_SOURCE_KEY] = self.src_path
            metadata[self.METADATA_SOURCE_KEY] = self.source
            metadata[self.METADATA_ARTIFACT_URI_KEY] = self.uri
            if self.get_target_path():
                metadata[self.METADATA_ARTIFACT_TARGET_PATH_KEY] = (
                    self.get_target_path()
                )

            for idx, text in enumerate(texts):
                metadata[self.METADATA_CHUNK_KEY] = str(idx)
                doc = Document(
                    page_content=text,
                    metadata=metadata,
                )
                results.append(doc)
        return results

    def collection_add(self, collection_id: str) -> None:
        self.spec.collections.add(collection_id)

    def collection_remove(self, collection_id: str) -> None:
        return self.spec.collections.discard(collection_id)
