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
import typing

import mlrun
from mlrun.artifacts import Artifact, ArtifactSpec
from mlrun.utils import StorePrefix, logger

MAX_PROMPT_LENGTH = 1000


class LLMPromptArtifact(Artifact):
    """
    LLM Prompt Artifact

    This artifact is used to store and manage LLM prompts.
    Stores the prompt string/path and a link to the related model artifact.
    """

    kind = "llm-prompt"
    _store_prefix = StorePrefix.LLMPrompt

    def read_prompt(self) -> str:
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


class LLMPromptArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "prompt_string",
        "prompt_legend",
        "generation_configuration",
        "description",
    ]

    def __init__(
        self,
        target_path=None,
        viewer=None,
        is_inline=False,
        format=None,
        size=None,
        db_key=None,
        extra_data=None,
        body=None,
        unpackaging_instructions: typing.Optional[dict] = None,
        parent_uri: typing.Optional[str] = None,
        prompt_string: typing.Optional[str] = None,
        prompt_path: typing.Optional[str] = None,
        prompt_legend: typing.Optional[dict] = None,
        generation_configuration: typing.Optional[dict] = None,
        description: typing.Optional[str] = None,
    ):
        if prompt_string and len(prompt_string) > MAX_PROMPT_LENGTH:
            logger.info("prompt_string is too long, creating a temp file")
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".txt"
            )
            temp_file.write(prompt_string)
            temp_file.flush()
            temp_file.close()
            prompt_path = temp_file.name
            prompt_string = None

        super().__init__(
            src_path=prompt_path,
            target_path=target_path,
            viewer=viewer,
            is_inline=is_inline,
            format=format,
            size=size,
            db_key=db_key,
            extra_data=extra_data,
            body=body,
            unpackaging_instructions=unpackaging_instructions,
            parent_uri=parent_uri,
        )

        self.prompt_string = prompt_string
        self.prompt_legend = prompt_legend
        self.generation_configuration = generation_configuration
        self.description = description

    @property
    def model_uri(self):
        return self.parent_uri

    def get_body(self):
        return self.prompt_string
