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

import pydantic


class ImageBuilder(pydantic.BaseModel):
    functionSourceCode: typing.Optional[str] = None  # noqa: N815
    codeEntryType: typing.Optional[str] = None  # noqa: N815
    codeEntryAttributes: typing.Optional[str] = None  # noqa: N815
    source: typing.Optional[str] = None
    code_origin: typing.Optional[str] = None
    origin_filename: typing.Optional[str] = None
    image: typing.Optional[str] = None
    base_image: typing.Optional[str] = None
    commands: typing.Optional[list] = None
    extra: typing.Optional[str] = None
    extra_args: typing.Optional[dict] = None
    builder_env: typing.Optional[dict] = None
    secret: typing.Optional[str] = None
    registry: typing.Optional[str] = None
    load_source_on_run: typing.Optional[bool] = None
    with_mlrun: typing.Optional[bool] = None
    auto_build: typing.Optional[bool] = None
    build_pod: typing.Optional[str] = None
    requirements: typing.Optional[list] = None
    source_code_target_dir: typing.Optional[str] = None

    class Config:
        extra = pydantic.Extra.allow


class LabelsModel(pydantic.BaseModel):
    """
    A model for handling labels.

    This class accepts either a dictionary or a list for filtering by labels.

    :param labels:
            - If a dictionary is provided, it should be in the format
              {'label_name': 'value'}. This will be converted to a list of strings
              in the format 'label_name=value'.
            - If a list is provided, all items must be strings. Each string can either
              be a simple label name (e.g., 'label1') or a key-value pair in the format
              'label=value'.
            - If no labels are specified, the default is an empty list.
    """

    labels: typing.Optional[typing.Union[dict[str, str], list[str]]]

    @pydantic.validator("labels", pre=True)
    @classmethod
    def validate_labels(cls, labels) -> list[str]:
        if labels is None:
            return []

        if isinstance(labels, list):
            if not all(isinstance(item, str) for item in labels):
                raise ValueError("All items in the list must be strings.")
            return labels

        if isinstance(labels, dict):
            return [f"{key}={value}" for key, value in labels.items()]

        raise ValueError("Invalid labels format. Must be a dict or a list of strings.")
