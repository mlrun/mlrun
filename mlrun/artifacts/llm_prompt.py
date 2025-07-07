# Copyright 2025 Iguazio
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
import tempfile
from typing import Optional, Union

import mlrun
import mlrun.artifacts.model as model_art
import mlrun.common
from mlrun.artifacts import Artifact, ArtifactMetadata, ArtifactSpec
from mlrun.utils import StorePrefix, logger

MAX_PROMPT_LENGTH = 1024


class LLMPromptArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "prompt_string",
        "prompt_legend",
        "model_configuration",
        "description",
    ]

    def __init__(
        self,
        model_artifact: Union[model_art.ModelArtifact, str] = None,
        prompt_string: Optional[str] = None,
        prompt_path: Optional[str] = None,
        prompt_legend: Optional[dict] = None,
        model_configuration: Optional[dict] = None,
        description: Optional[str] = None,
        target_path: Optional[str] = None,
        **kwargs,
    ):
        if prompt_string and prompt_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot specify both 'prompt_string' and 'prompt_path'"
            )

        super().__init__(
            src_path=prompt_path,
            target_path=target_path,
            parent_uri=model_artifact.uri
            if isinstance(model_artifact, model_art.ModelArtifact)
            else model_artifact,
            body=prompt_string,
            **kwargs,
        )

        self.prompt_string = prompt_string
        self.prompt_legend = prompt_legend
        self.model_configuration = model_configuration
        self.description = description
        self._model_artifact = None

    @property
    def model_uri(self):
        return self.parent_uri


class LLMPromptArtifact(Artifact):
    """
    LLM Prompt Artifact

    This artifact is used to store and manage LLM prompts.
    Stores the prompt string/path and a link to the related model artifact.
    """

    kind = mlrun.common.schemas.ArtifactCategories.llm_prompt
    _store_prefix = StorePrefix.LLMPrompt

    def __init__(
        self,
        key: Optional[str] = None,
        project: Optional[str] = None,
        model_artifact: Union[
            model_art.ModelArtifact, str
        ] = None,  # TODO support partial model uri
        prompt_string: Optional[str] = None,
        prompt_path: Optional[str] = None,
        prompt_legend: Optional[dict] = None,
        model_configuration: Optional[dict] = None,
        description: Optional[str] = None,
        target_path=None,
        **kwargs,
    ):
        llm_prompt_spec = LLMPromptArtifactSpec(
            prompt_string=prompt_string,
            prompt_path=prompt_path,
            prompt_legend=prompt_legend,
            model_artifact=model_artifact,
            model_configuration=model_configuration,
            target_path=target_path,
            description=description,
        )

        llm_metadata = ArtifactMetadata(
            key=key,
            project=project or "",
        )

        super().__init__(spec=llm_prompt_spec, metadata=llm_metadata, **kwargs)

    @property
    def spec(self) -> LLMPromptArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: LLMPromptArtifactSpec):
        self._spec = self._verify_dict(spec, "spec", LLMPromptArtifactSpec)

    @property
    def model_artifact(self) -> Optional[model_art.ModelArtifact]:
        """
        Get the model artifact linked to this prompt artifact.
        """
        if self.spec._model_artifact:
            return self.spec._model_artifact
        if self.spec.model_uri:
            self.spec._model_artifact, _ = (
                mlrun.datastore.store_manager.get_store_artifact(self.spec.model_uri)
            )
            return self.spec._model_artifact
        return None

    def read_prompt(self) -> Optional[str]:
        """
        Read the prompt string from the artifact.
        """
        if self.spec.prompt_string:
            return self.spec.prompt_string
        if self.spec.target_path:
            with mlrun.datastore.store_manager.object(url=self.spec.target_path).open(
                mode="r"
            ) as p_file:
                return p_file.read()

    def before_log(self):
        """
        Prepare the artifact before logging.
        This method is called before the artifact is logged.
        """
        if self.spec.prompt_string and len(self.spec.prompt_string) > MAX_PROMPT_LENGTH:
            logger.debug(
                "Prompt string exceeds maximum length, saving to a temporary file."
            )
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".txt"
            ) as temp_file:
                temp_file.write(self.spec.prompt_string)
            self.spec.src_path = temp_file.name
            self.spec.prompt_string = None
            self._src_is_temp = True

        super().before_log()
