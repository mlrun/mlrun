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
import json
import tempfile
from collections import defaultdict
from typing import Optional, Union

import mlrun
import mlrun.artifacts.model as model_art
import mlrun.common.schemas
from mlrun.artifacts import Artifact, ArtifactMetadata, ArtifactSpec
from mlrun.utils import StorePrefix, logger

MAX_PROMPT_LENGTH = 1024


class LLMPromptArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "prompt_template",
        "prompt_legend",
        "invocation_config",
        "description",
    ]
    PROMPT_TEMPLATE_KEYS = ("content", "role")
    PROMPT_LEGENDS_KEYS = ("field", "description")

    def __init__(
        self,
        model_artifact: Union[model_art.ModelArtifact, str] = None,
        prompt_template: Optional[list[dict]] = None,
        prompt_path: Optional[str] = None,
        prompt_legend: Optional[dict] = None,
        invocation_config: Optional[dict] = None,
        description: Optional[str] = None,
        target_path: Optional[str] = None,
        **kwargs,
    ):
        if prompt_template and prompt_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot specify both 'prompt_template' and 'prompt_path'"
            )
        if prompt_legend:
            self._verify_prompt_legend(prompt_legend)
        if prompt_path:
            self._verify_prompt_path(prompt_path)
        if prompt_template:
            self._verify_prompt_template(prompt_template)
        super().__init__(
            src_path=prompt_path,
            target_path=target_path,
            parent_uri=model_artifact.uri
            if isinstance(model_artifact, model_art.ModelArtifact)
            else model_artifact,
            format=kwargs.pop("format", "") or "json",
            **kwargs,
        )

        self.prompt_template = prompt_template
        self.prompt_legend = prompt_legend
        if invocation_config is not None and not isinstance(invocation_config, dict):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "LLMPromptArtifact invocation_config must be a dictionary or None"
            )
        self.invocation_config = invocation_config or {}
        self.description = description
        self._model_artifact = (
            model_artifact
            if isinstance(model_artifact, model_art.ModelArtifact)
            else None
        )

    def _verify_prompt_template(self, prompt_template):
        if not (
            isinstance(prompt_template, list)
            and all(isinstance(item, dict) for item in prompt_template)
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Expected prompt_template to be a list of dicts"
            )
        for message in prompt_template:
            if set(key.lower() for key in message.keys()) != set(
                self.PROMPT_TEMPLATE_KEYS
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Expected prompt_template to contain dicts with keys "
                    f"{self.PROMPT_TEMPLATE_KEYS}, got {message.keys()}"
                )
            keys_to_pop = []
            for key in message.keys():
                if isinstance(key, str):
                    if not key.islower():
                        message[key.lower()] = message[key]
                        keys_to_pop.append(key)
                else:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Expected prompt_template to contain dict that only"
                        f" has str keys got {key} of type {type(key)}"
                    )
            for key_to_pop in keys_to_pop:
                message.pop(key_to_pop)

    @property
    def model_uri(self):
        return self.parent_uri

    @staticmethod
    def _verify_prompt_legend(prompt_legend: dict):
        if prompt_legend is None:
            return True
        for place_holder, body_map in prompt_legend.items():
            if isinstance(body_map, dict):
                if body_map.get("field") is None:
                    body_map["field"] = place_holder
                body_map["description"] = body_map.get("description")
                if diff := set(body_map.keys()) - set(
                    LLMPromptArtifactSpec.PROMPT_LEGENDS_KEYS
                ):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "prompt_legend values must contain only 'field' and "
                        f"'description' keys, got extra fields: {diff}"
                    )
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Wrong prompt_legend format, {place_holder} is not mapped to dict"
                )

    @staticmethod
    def _verify_prompt_path(prompt_path: str):
        with mlrun.datastore.store_manager.object(prompt_path).open(mode="r") as p_file:
            try:
                json.load(p_file)
            except json.JSONDecodeError:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Failed on decoding str in path "
                    f"{prompt_path} expected file to contain a "
                    f"json format."
                )

    def get_body(self):
        if self.prompt_template:
            return json.dumps(self.prompt_template)
        else:
            return None


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
        prompt_template: Optional[list[dict]] = None,
        prompt_path: Optional[str] = None,
        prompt_legend: Optional[dict] = None,
        invocation_config: Optional[dict] = None,
        description: Optional[str] = None,
        target_path=None,
        **kwargs,
    ):
        llm_prompt_spec = LLMPromptArtifactSpec(
            prompt_template=prompt_template,
            prompt_path=prompt_path,
            prompt_legend=prompt_legend,
            model_artifact=model_artifact,
            invocation_config=invocation_config,
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

    def read_prompt(self) -> Optional[Union[str, list[dict]]]:
        """
        Read the prompt json from the artifact or if provided prompt template.
        @:param as_str: True to return the prompt string or a list of dicts.
        @:return prompt string or list of dicts
        """
        if self.spec.prompt_template:
            return self.spec.prompt_template
        if self.spec.target_path:
            with mlrun.datastore.store_manager.object(url=self.spec.target_path).open(
                mode="r"
            ) as p_file:
                try:
                    return json.load(p_file)
                except json.JSONDecodeError:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Failed on decoding str in path "
                        f"{self.spec.target_path} expected file to contain a "
                        f"json format."
                    )

    def before_log(self):
        """
        Prepare the artifact before logging.
        This method is called before the artifact is logged.
        """
        if (
            self.spec.prompt_template
            and len(str(self.spec.prompt_template)) > MAX_PROMPT_LENGTH
        ):
            logger.debug(
                "Prompt string exceeds maximum length, saving to a temporary file."
            )
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".json"
            ) as temp_file:
                temp_file.write(json.dumps(self.spec.prompt_template))
            self.spec.src_path = temp_file.name
            self.spec.prompt_template = None
            self._src_is_temp = True
        super().before_log()


class PlaceholderDefaultDict(defaultdict):
    def __missing__(self, key):
        return f"{{{key}}}"
