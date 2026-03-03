# Copyright 2023 Iguazio
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

import typing

import pydantic.v1

import mlrun.errors


class ImageBuilder(pydantic.v1.BaseModel):
    functionSourceCode: str | None = None  # noqa: N815
    codeEntryType: str | None = None  # noqa: N815
    codeEntryAttributes: str | None = None  # noqa: N815
    source: str | None = None
    code_origin: str | None = None
    origin_filename: str | None = None
    image: str | None = None
    base_image: str | None = None
    commands: list | None = None
    extra: str | None = None
    extra_args: dict | None = None
    builder_env: dict | None = None
    secret: str | None = None
    registry: str | None = None
    load_source_on_run: bool | None = None
    with_mlrun: bool | None = None
    auto_build: bool | None = None
    build_pod: str | None = None
    requirements: list | None = None
    source_code_target_dir: str | None = None

    class Config:
        extra = pydantic.v1.Extra.allow


class LabelsModel(pydantic.v1.BaseModel):
    """
    This class accepts either a dictionary, a list, or a string for filtering by labels.

    :param labels:
        - If a dictionary is provided, it should be in the format {'label_name': 'value'}.
          The values can also be `None`, which will result in the format 'label_name' (without a value).
          This will be converted to a list of strings in the format 'label_name=value'.
        - If a list is provided, all items must be strings. Each string can either
          be a simple label name (e.g., 'label1') or a key-value pair in the format
          'label=value'.
        - If a string is provided, it should be a comma-separated list of labels
          (e.g., 'label1,label2').
        - If no labels are specified, the default is an empty list.
    """

    labels: typing.Union[str, dict[str, str | None], list[str]] | None

    @pydantic.v1.validator("labels")
    @classmethod
    def validate(cls, labels) -> list[str]:
        if labels is None:
            return []

        # If labels is a string, split it by commas
        if isinstance(labels, str):
            return [label.strip() for label in labels.split(",") if label.strip()]

        if isinstance(labels, list):
            if not all(isinstance(item, str) for item in labels):
                raise mlrun.errors.MLRunValueError(
                    "All items in the list must be strings."
                )
            return labels

        if isinstance(labels, dict):
            return [
                f"{key}={value}" if value is not None else key
                for key, value in labels.items()
            ]

        raise mlrun.errors.MLRunValueError(
            "Invalid labels format. Must be a string, dictionary of strings, or a list of strings."
        )
